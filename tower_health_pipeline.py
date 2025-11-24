"""
tower_health_pipeline.py
Extract tower positions and health values from Clash Royale frames using YOLO + PaddleOCR.
"""
import cv2
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from ultralytics import YOLO
import easyocr

HEALTH_BAR_ROIS_PATH = Path("/home/ostikar/MyProjects/CS541/ClashRoyale/data/towers3cls/bar_rois.json")

@dataclass
class TowerDetection:
    tower_type: str  # 'king' or 'princess'
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    health: Optional[int]
    health_confidence: Optional[float]
    position: str  # 'top' or 'bottom' (based on y-coordinate)
    side: Optional[str]  # 'left', 'right', 'center' (for princess towers)

class TowerHealthExtractor:
    def __init__(self, 
                 yolo_weights: str = 'runs/detect/towers5/weights/best.pt',
                 use_gpu: bool = True,
                 debug: bool = False):
        """
        Initialize tower detection and health extraction pipeline.
        
        Args:
            yolo_weights: Path to trained YOLO model
            use_gpu: Use GPU for OCR if available
            debug: Save intermediate debug images
        """
        print("Loading YOLO model...")
        self.model = YOLO(yolo_weights)
        
        print("Initializing EasyOCR...")
        # Set custom directories to avoid permission issues
        easyocr_dir = Path.home() / '.EasyOCR'
        self.ocr = easyocr.Reader(
            ['en'],
            gpu=use_gpu,
            verbose=False,
            model_storage_directory=str(easyocr_dir / 'model'),
            user_network_directory=str(easyocr_dir / 'user_network')
        )
        
        self.debug = debug
        self.class_names = {0: 'king', 1: 'princess'}
        
        # Load health bar ROIs
        self.health_bar_rois = self.load_health_bar_rois()
    
    def load_health_bar_rois(self) -> Dict[str, List[float]]:
        """Load predefined health bar ROIs from JSON file."""
        if HEALTH_BAR_ROIS_PATH.exists():
            with open(HEALTH_BAR_ROIS_PATH, 'r') as f:
                return json.load(f)
        return {}
        
    def get_health_roi(self, tower_type: str, position: str, side: str, img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Get health bar ROI from predefined coordinates.
        
        Args:
            tower_type: 'king' or 'princess'
            position: 'top' or 'bottom'
            side: 'left', 'right', or 'center'
            img_shape: (height, width) of image
            
        Returns:
            (x1, y1, x2, y2) health bar ROI in pixel coordinates
        """
        h, w = img_shape
        
        # Map tower identification to ROI key
        if tower_type == 'king':
            roi_key = f"king_{position}_bar"
        else:  # princess
            side_map = {'left': 'l', 'right': 'r'}
            roi_key = f"princess_{position}_{side_map.get(side, 'l')}_bar"
        
        if roi_key in self.health_bar_rois:
            roi = self.health_bar_rois[roi_key]
            # Convert normalized to pixel coordinates
            return (int(roi[0] * w), int(roi[1] * h), 
                   int(roi[2] * w), int(roi[3] * h))
        
        # Fallback to empty ROI if not found
        return (0, 0, 0, 0)
    
    def preprocess_health_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess health text region for OCR using color segmentation.
        
        Args:
            crop: BGR image crop
            
        Returns:
            Preprocessed binary image with isolated text
        """
        if crop.size == 0:
            return crop

        # Upscale for better OCR performance on small text
        scale_factor = 4
        h, w = crop.shape[:2]
        upscaled_crop = cv2.resize(crop, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LANCZOS4)

        # Convert to HSV color space
        hsv = cv2.cvtColor(upscaled_crop, cv2.COLOR_BGR2HSV)
        
        # Define a more restrictive range for white color (the numbers)
        # Focusing on high value, low saturation pixels
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Create a mask for the white text
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Advanced morphological cleaning
        # 1. Open to remove small noise artifacts
        open_kernel = np.ones((2,2), np.uint8)
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        # 2. Close to fill gaps within the numbers
        close_kernel = np.ones((3,3), np.uint8)
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        
        # Invert colors: OCR can sometimes perform better on dark text on a white background
        inverted_mask = cv2.bitwise_not(closed_mask)
        
        return inverted_mask
    
    def extract_health_number(self, crop: np.ndarray, tower_id: str = "") -> Tuple[Optional[int], Optional[float]]:
        """
        Extract numeric health value using OCR.
        
        Args:
            crop: Preprocessed image crop
            tower_id: Identifier for debug logging
            
        Returns:
            (health_value, confidence) or (None, None)
        """
        if crop.size == 0:
            return None, None
            
        try:
            # EasyOCR returns list of (bbox, text, confidence)
            result = self.ocr.readtext(crop)
            
            if not result:
                return None, None
            
            # Concatenate all detected text
            texts = []
            confidences = []
            for detection in result:
                # detection is (bbox, text, confidence)
                if len(detection) >= 3:
                    _, text, conf = detection
                    texts.append(str(text))
                    confidences.append(float(conf))
            
            if not texts:
                return None, None
            
            full_text = ''.join(texts)
            
            # Extract digits only
            digits = ''.join(c for c in full_text if c.isdigit())
            
            if not digits:
                return None, None
            
            health = int(digits)
            avg_conf = np.mean(confidences) if confidences else 0.0
            
            # Sanity check (Clash Royale tower health range)
            # Allow: 1-14 (king level at full health), 100-10000 (actual health values)
            # Reject very small numbers that are likely misreads
            if health < 1 or (health > 50 and health < 100) or health > 10000:
                if self.debug:
                    print(f"  [WARNING] {tower_id}: Health {health} outside valid range (text='{full_text}')")
                return None, None
            
            # For king towers with small numbers (1-14), it's likely the level indicator
            # In Clash Royale, kings start at full health with no visible bar
            # We'll return None to indicate no health data available
            if tower_id.startswith('king') and 1 <= health <= 20:
                if self.debug:
                    print(f"  [INFO] {tower_id}: Detected king level {health}, no health bar visible")
                return None, None
            
            return health, avg_conf
            
        except Exception as e:
            if self.debug:
                print(f"  [ERROR] {tower_id} OCR failed: {e}")
            return None, None
    
    def classify_tower_position(self, bbox: List[float], img_width: int, img_height: int) -> Tuple[str, Optional[str]]:
        """
        Classify tower as top/bottom and left/right/center.
        
        Args:
            bbox: [x1, y1, x2, y2]
            img_width: Image width
            img_height: Image height
            
        Returns:
            (position, side) e.g., ('top', 'left')
        """
        x1, y1, x2, y2 = bbox
        center_y = (y1 + y2) / 2
        center_x = (x1 + x2) / 2
        
        # Top or bottom half
        position = 'top' if center_y < img_height / 2 else 'bottom'
        
        # Left, center, or right third
        if center_x < img_width / 3:
            side = 'left'
        elif center_x > 2 * img_width / 3:
            side = 'right'
        else:
            side = 'center'
        
        return position, side
    
    def process_frame(self, frame_path: Path, save_debug: bool = False) -> List[TowerDetection]:
        """
        Process a single frame to detect towers and extract health.
        
        Args:
            frame_path: Path to input image
            save_debug: Save annotated debug image
            
        Returns:
            List of TowerDetection objects
        """
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"[ERROR] Could not read {frame_path}")
            return []
        
        h, w = frame.shape[:2]
        
        # YOLO detection
        results = self.model.predict(frame, verbose=False, conf=0.4)[0]
        
        detections = []
        debug_frame = frame.copy() if (save_debug or self.debug) else None
        
        for idx, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            tower_type = self.class_names.get(cls_id, 'unknown')
            position, side = self.classify_tower_position(xyxy, w, h)
            tower_id = f"{tower_type}_{position}_{side}"
            
            # Get health region using predefined ROIs
            tx1, ty1, tx2, ty2 = self.get_health_roi(tower_type, position, side, (h, w))
            health_crop = frame[ty1:ty2, tx1:tx2]
            
            if self.debug and health_crop.size == 0:
                print(f"  [WARNING] {tower_id}: Empty health crop! ROI coords: ({tx1},{ty1})-({tx2},{ty2})")
            
            # Try OCR on original crop first (health bars have distinct colors)
            health, health_conf = self.extract_health_number(health_crop, tower_id)
            
            # If that fails, try preprocessed version
            if health is None:
                processed_crop = self.preprocess_health_crop(health_crop)
                health, health_conf = self.extract_health_number(processed_crop, tower_id)
                
                # Save both versions for debugging
                if save_debug or self.debug:
                    if health_crop.size > 0:
                        orig_path = frame_path.parent / f"health_orig_{tower_id}_{frame_path.stem}.png"
                        cv2.imwrite(str(orig_path), health_crop)
                    if processed_crop.size > 0:
                        proc_path = frame_path.parent / f"health_proc_{tower_id}_{frame_path.stem}.png"
                        cv2.imwrite(str(proc_path), processed_crop)
            else:
                # Save successful original crop
                if save_debug or self.debug:
                    if health_crop.size > 0:
                        orig_path = frame_path.parent / f"health_orig_{tower_id}_{frame_path.stem}.png"
                        cv2.imwrite(str(orig_path), health_crop)
            
            detection = TowerDetection(
                tower_type=tower_type,
                bbox=xyxy.tolist(),
                confidence=conf,
                health=health,
                health_confidence=health_conf,
                position=position,
                side=side
            )
            detections.append(detection)
            
            # Debug visualization
            if debug_frame is not None:
                x1, y1, x2, y2 = map(int, xyxy)
                # Tower box (green)
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Health ROI (blue)
                cv2.rectangle(debug_frame, (tx1, ty1), (tx2, ty2), (255, 0, 0), 1)
                # Label
                label = f"{tower_type} {health if health else '?'}"
                cv2.putText(debug_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save_debug and debug_frame is not None:
            debug_path = frame_path.parent / f"debug_{frame_path.name}"
            cv2.imwrite(str(debug_path), debug_frame)
        
        return detections
    
    def process_game(self, game_dir: Path, output_json: Optional[Path] = None) -> Dict:
        """
        Process all frames in a game directory.
        
        Args:
            game_dir: Path to game_X directory containing images/
            output_json: Optional path to save results JSON
            
        Returns:
            Dictionary with frame-by-frame results
        """
        images_dir = game_dir / 'images'
        if not images_dir.exists():
            print(f"[ERROR] No images directory in {game_dir}")
            return {}
        
        frames = sorted(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')))
        print(f"Processing {len(frames)} frames from {game_dir.name}...")
        
        results = {'game': game_dir.name, 'frames': {}}
        
        for frame_path in frames:
            detections = self.process_frame(frame_path)
            results['frames'][frame_path.name] = [asdict(d) for d in detections]
        
        # Summary stats
        total_towers = sum(len(v) for v in results['frames'].values())
        health_extracted = sum(
            sum(1 for d in v if d['health'] is not None) 
            for v in results['frames'].values()
        )
        
        results['summary'] = {
            'total_frames': len(frames),
            'total_detections': total_towers,
            'health_extracted': health_extracted,
            'ocr_success_rate': health_extracted / total_towers if total_towers > 0 else 0
        }
        
        print(f"  Towers detected: {total_towers}")
        print(f"  Health extracted: {health_extracted}/{total_towers} ({results['summary']['ocr_success_rate']:.1%})")
        
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Results saved to {output_json}")
        
        return results


def main():
    """Example usage and validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract tower health from Clash Royale frames')
    parser.add_argument('--game-dir', type=str, 
                       default='/home/ostikar/MyProjects/CS541/ClashRoyale/data/arena_01/game_1',
                       help='Path to game directory')
    parser.add_argument('--weights', type=str,
                       default='runs/detect/towers5/weights/best.pt',
                       help='Path to YOLO weights')
    parser.add_argument('--output', type=str, default='tower_health_results.json',
                       help='Output JSON file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with visualizations')
    parser.add_argument('--sample-frames', type=int, default=None,
                       help='Process only first N frames (for testing)')
    
    args = parser.parse_args()
    
    extractor = TowerHealthExtractor(
        yolo_weights=args.weights,
        use_gpu=False,  # Use CPU for OCR to avoid cuDNN issues
        debug=args.debug
    )
    
    game_dir = Path(args.game_dir)
    
    if args.sample_frames:
        # Process sample frames with debug output
        images_dir = game_dir / 'images'
        frames = sorted(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')))[:args.sample_frames]
        print(f"\nProcessing {len(frames)} sample frames with debug visualization...\n")
        for frame_path in frames:
            print(f"Frame: {frame_path.name}")
            detections = extractor.process_frame(frame_path, save_debug=True)
            for d in detections:
                print(f"  {d.tower_type} ({d.position}-{d.side}): health={d.health} conf={d.confidence:.2f}")
            print()
    else:
        # Process full game
        results = extractor.process_game(game_dir, output_json=Path(args.output))
        
        print("\n" + "="*60)
        print("Pipeline Complete!")
        print(f"OCR Success Rate: {results['summary']['ocr_success_rate']:.1%}")
        print("="*60)


if __name__ == '__main__':
    main()
"""
Detect troop placement locations using template matching on the placement clock icon.
The clock appears briefly when a troop is placed in Clash Royale.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json


class PlacementDetector:
    """Detect troop placements using template matching on clock icons."""
    
    def __init__(self, template_path: Optional[str] = None, threshold: float = 0.7):
        """
        Initialize the placement detector.
        
        Args:
            template_path: Path to the clock template image. If None, will need to create one.
            threshold: Matching threshold (0-1). Higher = more strict matching.
        """
        self.threshold = threshold
        self.template = None
        self.template_w = None
        self.template_h = None
        
        if template_path and Path(template_path).exists():
            self.load_template(template_path)
    
    def load_template(self, template_path: str):
        """Load the clock template image."""
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise ValueError(f"Could not load template from {template_path}")
        self.template_h, self.template_w = self.template.shape
        print(f"Loaded template: {self.template_w}x{self.template_h}")
    
    def create_template_from_roi(self, image_path: str, x: int, y: int, w: int, h: int, 
                                 save_path: str = "clock_template.png"):
        """
        Extract a clock region from an image to use as template.
        
        Args:
            image_path: Path to image containing a clock
            x, y: Top-left coordinates of the clock
            w, h: Width and height of the clock region
            save_path: Where to save the template
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Extract ROI
        roi = img[y:y+h, x:x+w]
        
        # Convert to grayscale
        self.template = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.template_h, self.template_w = self.template.shape
        
        # Save for future use
        cv2.imwrite(save_path, self.template)
        print(f"Created and saved template to {save_path}")
        print(f"Template size: {self.template_w}x{self.template_h}")
        
        return self.template
    
    def load_multiple_templates(self, template_dir: str):
        """
        Load multiple clock templates from a directory.
        Useful for handling different clock animation states.
        
        Args:
            template_dir: Directory containing template images (clock_*.png)
        """
        template_path = Path(template_dir)
        self.templates = []
        
        for img_path in sorted(template_path.glob("clock_*.png")):
            template = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.templates.append({
                    'image': template,
                    'width': template.shape[1],
                    'height': template.shape[0],
                    'name': img_path.name
                })
        
        if not self.templates:
            raise ValueError(f"No clock_*.png templates found in {template_dir}")
        
        print(f"Loaded {len(self.templates)} clock templates:")
        for t in self.templates:
            print(f"  - {t['name']} ({t['width']}x{t['height']})")
        
        # Set primary template for backward compatibility
        self.template = self.templates[0]['image']
        self.template_w = self.templates[0]['width']
        self.template_h = self.templates[0]['height']
    
    def detect_placements(self, image_path: str, 
                         method: int = cv2.TM_CCOEFF_NORMED,
                         multi_scale: bool = False,
                         scales: List[float] = None,
                         use_all_templates: bool = True) -> List[Tuple[int, int, float]]:
        """
        Detect placement locations in an image.
        
        Args:
            image_path: Path to the game frame image
            method: OpenCV template matching method
            multi_scale: Whether to try multiple scales (usually not needed at fixed FPS)
            scales: List of scales to try (default: [0.9, 1.0, 1.1])
            use_all_templates: If True and multiple templates loaded, tries all of them
        
        Returns:
            List of (x, y, confidence) tuples for detected placements
        """
        # Check if we have templates
        has_multiple = hasattr(self, 'templates') and len(self.templates) > 0
        
        if not has_multiple and self.template is None:
            raise ValueError("No template loaded. Call load_template() or load_multiple_templates() first.")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # For 10fps with 1-second clock, multi-scale usually not needed
        if scales is None:
            scales = [0.9, 1.0, 1.1] if multi_scale else [1.0]
        
        all_detections = []
        
        # Determine which templates to use
        if has_multiple and use_all_templates:
            templates_to_try = self.templates
        else:
            # Single template mode
            templates_to_try = [{
                'image': self.template,
                'width': self.template_w,
                'height': self.template_h,
                'name': 'single'
            }]
        
        # Try each template
        for template_info in templates_to_try:
            template_img = template_info['image']
            template_w = template_info['width']
            template_h = template_info['height']
            
            # Try multiple scales for this template
            for scale in scales:
                # Resize template
                if scale != 1.0:
                    w = int(template_w * scale)
                    h = int(template_h * scale)
                    template_scaled = cv2.resize(template_img, (w, h))
                else:
                    template_scaled = template_img
                    w, h = template_w, template_h
                
                # Skip if template is larger than image
                if w > gray.shape[1] or h > gray.shape[0]:
                    continue
                
                # Perform template matching
                result = cv2.matchTemplate(gray, template_scaled, method)
                
                # Find locations above threshold
                locations = np.where(result >= self.threshold)
                
                # Store detections with their confidence scores
                for pt in zip(*locations[::-1]):  # Switch x and y
                    confidence = result[pt[1], pt[0]]
                    # Store center point instead of top-left
                    center_x = pt[0] + w // 2
                    center_y = pt[1] + h // 2
                    all_detections.append((center_x, center_y, confidence, scale))
        
        # Apply Non-Maximum Suppression to remove duplicate detections
        detections = self._non_max_suppression(all_detections, overlap_threshold=30)
        
        return detections
    
    def _non_max_suppression(self, detections: List[Tuple[int, int, float, float]], 
                            overlap_threshold: int = 30) -> List[Tuple[int, int, float]]:
        """
        Remove overlapping detections, keeping only the one with highest confidence.
        
        Args:
            detections: List of (x, y, confidence, scale)
            overlap_threshold: Maximum distance (pixels) between detections to be considered duplicates
        
        Returns:
            Filtered list of (x, y, confidence)
        """
        if not detections:
            return []
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x[2], reverse=True)
        
        kept = []
        
        for det in detections:
            x, y, conf, scale = det
            
            # Check if this detection overlaps with any kept detection
            is_duplicate = False
            for kept_det in kept:
                kx, ky, kconf = kept_det
                distance = np.sqrt((x - kx)**2 + (y - ky)**2)
                
                if distance < overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append((x, y, conf))
        
        return kept
    
    def visualize_detections(self, image_path: str, detections: List[Tuple[int, int, float]], 
                            output_path: str = None, show: bool = False):
        """
        Visualize detected placements on the image.
        
        Args:
            image_path: Path to the game frame
            detections: List of (x, y, confidence) from detect_placements()
            output_path: Where to save the annotated image
            show: Whether to display the image
        """
        img = cv2.imread(image_path)
        
        for x, y, conf in detections:
            # Draw circle at placement location
            cv2.circle(img, (x, y), 10, (0, 255, 0), 2)
            # Draw crosshair
            cv2.line(img, (x-15, y), (x+15, y), (0, 255, 0), 2)
            cv2.line(img, (x, y-15), (x, y+15), (0, 255, 0), 2)
            # Add confidence text
            cv2.putText(img, f"{conf:.2f}", (x+15, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Saved visualization to {output_path}")
        
        if show:
            cv2.imshow("Placement Detections", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return img


def extract_clock_template_interactive(image_path: str, output_template: str = "clock_template.png"):
    """
    Interactive tool to extract a clock template from an image.
    Click and drag to select the clock region.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    clone = img.copy()
    roi_selected = False
    start_point = None
    end_point = None
    
    def click_and_crop(event, x, y, flags, param):
        nonlocal start_point, end_point, roi_selected, img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (x, y)
            roi_selected = False
        
        elif event == cv2.EVENT_MOUSEMOVE and start_point:
            img = clone.copy()
            cv2.rectangle(img, start_point, (x, y), (0, 255, 0), 2)
        
        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            roi_selected = True
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
    
    cv2.namedWindow("Select Clock Region")
    cv2.setMouseCallback("Select Clock Region", click_and_crop)
    
    print("Instructions:")
    print("1. Click and drag to select the clock icon")
    print("2. Press 's' to save the template")
    print("3. Press 'r' to reset selection")
    print("4. Press 'q' to quit without saving")
    
    while True:
        cv2.imshow("Select Clock Region", img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and roi_selected:
            # Extract and save the ROI
            x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
            x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
            
            roi = clone[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(output_template, gray_roi)
            
            print(f"\nTemplate saved to {output_template}")
            print(f"Region: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}")
            break
        
        elif key == ord('r'):
            img = clone.copy()
            roi_selected = False
            start_point = None
            end_point = None
        
        elif key == ord('q'):
            print("Cancelled.")
            break
    
    cv2.destroyAllWindows()


# Example usage functions
def example_create_template():
    """Example: Create a template from a sample image."""
    # First, use interactive tool to select the clock
    sample_image = "/home/ostikar/MyProjects/CS541/ClashRoyale/data/arena_01/game_01/images/frame_0000.png"
    extract_clock_template_interactive(sample_image, "clock_template.png")


def example_detect_placements():
    """Example: Detect placements in an image using the template."""
    detector = PlacementDetector(template_path="clock_template.png", threshold=0.7)
    
    # Detect in a frame
    test_image = "/home/ostikar/MyProjects/CS541/ClashRoyale/data/arena_01/game_01/images/frame_0100.png"
    detections = detector.detect_placements(test_image, multi_scale=True)
    
    print(f"Found {len(detections)} placements:")
    for i, (x, y, conf) in enumerate(detections):
        print(f"  {i+1}. Position: ({x}, {y}), Confidence: {conf:.3f}")
    
    # Visualize
    detector.visualize_detections(test_image, detections, 
                                  output_path="placement_detections.png",
                                  show=True)


def batch_detect_placements(image_dir: str, template_path: str = None,
                           template_dir: str = None,
                           output_json: str = "placement_detections.json",
                           threshold: float = 0.65):
    """
    Detect placements across multiple frames and save results.
    
    Args:
        image_dir: Directory containing game frames
        template_path: Path to single clock template (deprecated, use template_dir)
        template_dir: Directory containing multiple clock templates (recommended)
        output_json: Where to save detection results
        threshold: Detection threshold (0.65 recommended for 10fps data)
    """
    detector = PlacementDetector(threshold=threshold)
    
    # Load templates
    if template_dir:
        detector.load_multiple_templates(template_dir)
        print(f"Using multi-template mode with {len(detector.templates)} templates")
    elif template_path:
        detector.load_template(template_path)
        print(f"Using single template mode")
    else:
        raise ValueError("Provide either template_path or template_dir")
    
    image_dir = Path(image_dir)
    results = {}
    
    print(f"\nProcessing frames from {image_dir}...")
    image_files = sorted(image_dir.glob("*.png"))
    
    for i, img_path in enumerate(image_files):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(image_files)} frames...")
        
        detections = detector.detect_placements(str(img_path), multi_scale=False)
        
        if detections:
            results[img_path.name] = [
                {"x": int(x), "y": int(y), "confidence": float(conf)}
                for x, y, conf in detections
            ]
            print(f"  {img_path.name}: {len(detections)} placements detected")
    
    # Save results
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Saved results to {output_json}")
    print(f"Total frames processed: {len(image_files)}")
    print(f"Frames with placements: {len(results)}")
    total_placements = sum(len(dets) for dets in results.values())
    print(f"Total placements detected: {total_placements}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Create template:     python detect_placement.py create_template <image_path>")
        print("  Detect single:       python detect_placement.py detect <image_path> <template_path>")
        print("  Detect multi-tmpl:   python detect_placement.py detect <image_path> --templates <template_dir>")
        print("  Batch (single):      python detect_placement.py batch <image_dir> <template_path>")
        print("  Batch (multi-tmpl):  python detect_placement.py batch <image_dir> --templates <template_dir>")
        print("\nRecommended for 10fps data: Use multi-template mode with 5-10 templates")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create_template":
        if len(sys.argv) < 3:
            print("Error: Provide image path")
            sys.exit(1)
        extract_clock_template_interactive(sys.argv[2])
    
    elif command == "detect":
        if len(sys.argv) < 4:
            print("Error: Provide image_path and template_path or --templates <dir>")
            sys.exit(1)
        
        image_path = sys.argv[2]
        
        # Check if using multi-template mode
        if sys.argv[3] == "--templates" and len(sys.argv) >= 5:
            detector = PlacementDetector(threshold=0.65)
            detector.load_multiple_templates(sys.argv[4])
        else:
            detector = PlacementDetector(template_path=sys.argv[3], threshold=0.65)
        
        detections = detector.detect_placements(image_path)
        
        print(f"\nFound {len(detections)} placements:")
        for i, (x, y, conf) in enumerate(detections):
            print(f"  {i+1}. Position: ({x}, {y}), Confidence: {conf:.3f}")
        
        # Save visualization
        output = image_path.replace('.png', '_detections.png')
        detector.visualize_detections(image_path, detections, output_path=output)
    
    elif command == "batch":
        if len(sys.argv) < 4:
            print("Error: Provide image_dir and template_path or --templates <dir>")
            sys.exit(1)
        
        image_dir = sys.argv[2]
        
        # Check if using multi-template mode
        if sys.argv[3] == "--templates" and len(sys.argv) >= 5:
            batch_detect_placements(image_dir, template_dir=sys.argv[4])
        else:
            batch_detect_placements(image_dir, template_path=sys.argv[3])
    
    else:
        print(f"Unknown command: {command}")

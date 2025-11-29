"""
Fast clock template extraction tool for a full game.
Iterates through all frames and lets you click on clocks quickly.
Perfect for creating 50-100+ templates from actual gameplay.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import argparse


class ClockExtractor:
    def __init__(self, output_dir: str, template_size: int = 40, remove_background: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_size = template_size
        self.remove_background = remove_background
        # Find next available template number
        existing = list(self.output_dir.glob("clock_*.png"))
        if existing:
            numbers = []
            for f in existing:
                try:
                    num = int(f.stem.split('_')[1])
                    numbers.append(num)
                except (ValueError, IndexError):
                    pass
            self.template_count = max(numbers) if numbers else 0
        else:
            self.template_count = 0
        self.current_frame = None
        self.current_frame_path = None
        self.display_img = None
        self.click_point = None
        self.pending_template = None  # Store template before saving
        self.frame_index = 0
        self.total_frames = 0
        self.skip_frames = 1  # Can adjust to go faster
        
    def on_mouse(self, event, x, y, flags, param):
        """Handle mouse clicks - prepare template at click location"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
            self.prepare_template_at_point(x, y)
            
            # Show feedback with preview
            if self.pending_template is not None:
                cv2.circle(self.display_img, (x, y), self.template_size//2, (0, 255, 255), 2)
                cv2.putText(self.display_img, f"Press 'S' to save #{self.template_count + 1}", 
                           (x+self.template_size//2+5, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(self.display_img, "or click elsewhere", 
                           (x+self.template_size//2+5, y+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    def prepare_template_at_point(self, x: int, y: int):
        """Prepare template at click point (doesn't save yet)"""
        half_size = self.template_size // 2
        
        # Calculate bounds with bounds checking
        y1 = max(0, y - half_size)
        y2 = min(self.current_frame.shape[0], y + half_size)
        x1 = max(0, x - half_size)
        x2 = min(self.current_frame.shape[1], x + half_size)
        
        # Extract ROI
        roi = self.current_frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            self.pending_template = None
            return
        
        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply background removal if enabled
        if self.remove_background:
            gray_roi = self._remove_background(roi, gray_roi)
        
        # Store pending template
        self.pending_template = {
            'roi': gray_roi,
            'click_point': (x, y),
            'region': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'frame_index': self.frame_index
        }
    
    def save_pending_template(self):
        """Save the pending template to disk"""
        if self.pending_template is None:
            print("No template to save")
            return
        
        gray_roi = self.pending_template['roi']
        x, y = self.pending_template['click_point']
        region = self.pending_template['region']
        
        # Save template with next number
        self.template_count += 1
        template_path = self.output_dir / f"clock_{self.template_count:04d}.png"
        cv2.imwrite(str(template_path), gray_roi)
        
        # Save metadata
        metadata = {
            'template_id': self.template_count,
            'source_frame': str(self.current_frame_path.name),
            'frame_index': self.pending_template['frame_index'],
            'click_point': {'x': x, 'y': y},
            'region': region,
            'size': {'width': region['x2']-region['x1'], 'height': region['y2']-region['y1']},
            'background_removed': self.remove_background
        }
        
        meta_path = self.output_dir / f"clock_{self.template_count:04d}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Template {self.template_count} saved: {template_path.name}")
        
        # Show saved feedback
        cv2.circle(self.display_img, (x, y), self.template_size//2, (0, 255, 0), 2)
        cv2.putText(self.display_img, f"Saved #{self.template_count}!", 
                   (x+self.template_size//2+5, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Clear pending
        self.pending_template = None
    
    def _remove_background(self, roi_color, roi_gray):
        """Remove background using color-based masking to focus on clock"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        
        # Detect blue clock (both light and dark blue) - more aggressive
        blue_lower = np.array([90, 40, 40])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Also detect red/orange clocks (opponent) - more aggressive
        red_lower1 = np.array([0, 40, 40])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 40, 40])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Combine masks
        clock_mask = cv2.bitwise_or(blue_mask, red_mask)
        
        # More aggressive morphological operations to clean up and focus on clock
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clock_mask = cv2.morphologyEx(clock_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        clock_mask = cv2.morphologyEx(clock_mask, cv2.MORPH_DILATE, kernel, iterations=2)
        
        # Apply mask to gray image - set background to mid-gray (127)
        # This is better than black (0) which could match dark backgrounds
        masked_gray = np.full_like(roi_gray, 127, dtype=np.uint8)
        masked_gray[clock_mask > 0] = roi_gray[clock_mask > 0]
        
        return masked_gray
    
    def process_game(self, images_dir: str, start_frame: int = 0):
        """Process all frames in a game directory"""
        images_path = Path(images_dir)
        
        if not images_path.exists():
            print(f"Error: {images_dir} not found")
            return
        
        # Get all frames
        frame_files = sorted(images_path.glob("*.png"))
        self.total_frames = len(frame_files)
        
        if self.total_frames == 0:
            print(f"No PNG frames found in {images_dir}")
            return
        
        print("\n" + "="*70)
        print("FAST CLOCK EXTRACTION")
        print("="*70)
        print(f"Game directory: {images_dir}")
        print(f"Total frames: {self.total_frames}")
        print(f"Starting template count: {self.template_count}")
        print(f"Template size: {self.template_size}x{self.template_size} px")
        print(f"Background removal: {'ON' if self.remove_background else 'OFF'}")
        print("\nControls:")
        print("  LEFT CLICK: Select clock at cursor (shows preview)")
        print("  S: Save the selected clock template")
        print("  SPACE/→: Next frame")
        print("  ←: Previous frame")
        print("  +/-: Increase/decrease frame skip (for faster navigation)")
        print("  r: Rewind to frame 0")
        print("  q: Quit and save")
        print("="*70 + "\n")
        
        # Start from specified frame
        self.frame_index = start_frame
        
        window_name = "Click on clocks (Q to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.on_mouse)
        
        while True:
            # Load current frame
            if self.frame_index < 0:
                self.frame_index = 0
            if self.frame_index >= self.total_frames:
                self.frame_index = self.total_frames - 1
            
            frame_path = frame_files[self.frame_index]
            self.current_frame = cv2.imread(str(frame_path))
            self.current_frame_path = frame_path
            
            if self.current_frame is None:
                print(f"Warning: Could not load {frame_path}")
                self.frame_index += 1
                continue
            
            # Create display image
            self.display_img = self.current_frame.copy()
            
            # Add info overlay
            info_text = [
                f"Frame: {self.frame_index}/{self.total_frames} ({frame_path.name})",
                f"Templates: {self.template_count}",
                f"Skip: {self.skip_frames} frames"
            ]
            
            if self.pending_template is not None:
                info_text.append(">>> Press 'S' to save template <<<")
            
            y_offset = 30
            for text in info_text:
                # Highlight pending save message
                color = (0, 255, 255) if "Press 'S'" in text else (0, 255, 0)
                cv2.putText(self.display_img, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30
            
            # Add crosshair at cursor position
            # (OpenCV doesn't provide cursor position, so we just show the frame)
            
            cv2.imshow(window_name, self.display_img)
            
            # Handle key press (wait for very short time to allow mouse events)
            key = cv2.waitKey(1) & 0xFF
            
            # Continue if no key pressed
            if key == 255:
                continue
            
            if key == ord('q'):
                print(f"\n✓ Quit. Total templates created: {self.template_count}")
                break
            
            elif key == ord('s') or key == ord('S'):
                if self.pending_template is not None:
                    self.save_pending_template()
                    cv2.imshow(window_name, self.display_img)
                    cv2.waitKey(300)  # Brief pause to show saved feedback
                else:
                    print("No template selected. Click on a clock first.")
            
            elif key == ord(' ') or key == 83:  # Space or right arrow
                self.frame_index += self.skip_frames
                self.pending_template = None  # Clear pending on frame change
                
            elif key == 81:  # Left arrow
                self.frame_index -= self.skip_frames
                self.pending_template = None  # Clear pending on frame change
                
            elif key == ord('r'):
                self.frame_index = 0
                self.pending_template = None  # Clear pending on rewind
                print("Rewound to frame 0")
                
            elif key == ord('+') or key == ord('='):
                self.skip_frames = min(10, self.skip_frames + 1)
                print(f"Skip: {self.skip_frames} frames")
                
            elif key == ord('-') or key == ord('_'):
                self.skip_frames = max(1, self.skip_frames - 1)
                print(f"Skip: {self.skip_frames} frames")
            
            # Note: Removed auto-advance after clicking
            # User now controls navigation with arrow keys/space after saving with 's'
        
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print(f"EXTRACTION COMPLETE")
        print("="*70)
        print(f"Total templates: {self.template_count}")
        print(f"Output directory: {self.output_dir}")
        print("\nNext steps:")
        print(f"  1. Test detection:")
        print(f"     python create_placement_dataset.py ../hf_subset \\")
        print(f"       --templates {self.output_dir} \\")
        print(f"       --arenas arena_31 \\")
        print(f"       --game <game_id> \\")
        print(f"       --output test_results.csv")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fast clock template extraction from game frames"
    )
    parser.add_argument("images_dir", 
                       help="Directory containing frame images (e.g., ../hf_subset/arena_31/<game_id>/images)")
    parser.add_argument("--output", "-o", default="clock_templates",
                       help="Output directory for templates (default: clock_templates)")
    parser.add_argument("--size", "-s", type=int, default=30,
                       help="Template size in pixels (default: 30, try 25-35)")
    parser.add_argument("--start", type=int, default=0,
                       help="Start from specific frame number (default: 0)")
    parser.add_argument("--no-bg-removal", action="store_true",
                       help="Disable background removal (keep raw templates)")
    
    args = parser.parse_args()
    
    extractor = ClockExtractor(args.output, 
                               template_size=args.size,
                               remove_background=not args.no_bg_removal)
    extractor.process_game(args.images_dir, start_frame=args.start)


if __name__ == "__main__":
    main()

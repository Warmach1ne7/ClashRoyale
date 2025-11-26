"""
Fast manual clock template extractor.
Just click on clocks across different frames - no drag needed!
Optimized for quickly creating 20-30 templates.
"""

import cv2
import numpy as np
from pathlib import Path
import json


def fast_extract_templates(frames_dir: str, output_dir: str, 
                           template_size: int = 35,
                           max_templates: int = 30):
    """
    Fast template extraction - just click on clock centers.
    
    Args:
        frames_dir: Directory with game frames
        output_dir: Where to save templates
        template_size: Size of template region (pixels from center)
        max_templates: Stop after this many templates
    """
    frames_path = Path(frames_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get existing template count
    existing = list(output_path.glob("clock_template_*.png"))
    template_count = len(existing)
    
    if template_count > 0:
        print(f"Found {template_count} existing templates")
        start_num = template_count + 1
    else:
        start_num = 1
    
    # Get all frames, spread across the dataset
    all_frames = sorted(frames_path.glob("**/*.png"))
    
    if not all_frames:
        print(f"No PNG files found in {frames_dir}")
        return 0
    
    print(f"\n{'='*70}")
    print(f"FAST CLOCK TEMPLATE EXTRACTOR")
    print(f"{'='*70}")
    print(f"Total frames available: {len(all_frames)}")
    print(f"Target templates: {max_templates}")
    print(f"Current templates: {template_count}")
    print(f"\nInstructions:")
    print(f"  1. Click on the CENTER of each clock you see")
    print(f"  2. Template will be extracted automatically ({template_size}x{template_size} px)")
    print(f"  3. Press SPACE to move to next frame")
    print(f"  4. Press 'u' to undo last template")
    print(f"  5. Press 'q' to quit")
    print(f"\nTip: Look for clocks at different animation stages")
    print(f"{'='*70}\n")
    
    # Sample frames evenly if we have many
    if len(all_frames) > 100:
        step = len(all_frames) // 100
        frame_list = [all_frames[i] for i in range(0, len(all_frames), step)]
    else:
        frame_list = all_frames
    
    current_img = None
    display_img = None
    frame_idx = 0
    clicks_this_frame = []
    saved_templates = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_img, display_img, template_count, clicks_this_frame, saved_templates
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Extract template centered at click
            half = template_size // 2
            
            # Bounds check
            if (x - half < 0 or x + half >= current_img.shape[1] or
                y - half < 0 or y + half >= current_img.shape[0]):
                print(f"  ⚠ Click too close to edge, skipping")
                return
            
            # Extract region
            roi = current_img[y-half:y+half, x-half:x+half]
            
            if roi.shape[0] != template_size or roi.shape[1] != template_size:
                print(f"  ⚠ Invalid region size, skipping")
                return
            
            # Convert to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Save template
            template_count += 1
            template_path = output_path / f"clock_template_{template_count:02d}.png"
            cv2.imwrite(str(template_path), gray_roi)
            
            # Save metadata
            metadata = {
                'template_id': template_count,
                'source_frame': str(param['frame_path']),
                'center': {'x': x, 'y': y},
                'size': template_size
            }
            meta_path = output_path / f"clock_template_{template_count:02d}.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            saved_templates.append((template_count, x, y, template_path))
            clicks_this_frame.append((x, y))
            
            # Draw marker on display
            cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(display_img, (x, y), half, (0, 255, 0), 2)
            cv2.putText(display_img, str(template_count), (x+20, y-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            print(f"  ✓ Template {template_count} saved: {template_path.name}")
            
            # Update progress
            remaining = max_templates - template_count
            if remaining > 0:
                print(f"    Progress: {template_count}/{max_templates} ({remaining} more needed)")
    
    window_name = "Click on clocks | SPACE=next frame | U=undo | Q=quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    while frame_idx < len(frame_list) and template_count < max_templates:
        frame_path = frame_list[frame_idx]
        current_img = cv2.imread(str(frame_path))
        
        if current_img is None:
            frame_idx += 1
            continue
        
        display_img = current_img.copy()
        clicks_this_frame = []
        
        # Set up mouse callback with frame info
        cv2.setMouseCallback(window_name, mouse_callback, 
                            {'frame_path': frame_path})
        
        # Display frame info
        info_text = f"Frame {frame_idx+1}/{len(frame_list)} | {frame_path.name} | Templates: {template_count}/{max_templates}"
        cv2.putText(display_img, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        while True:
            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Next frame
                print(f"\nMoving to next frame ({clicks_this_frame.__len__()} clocks marked)\n")
                break
            
            elif key == ord('u'):  # Undo last template
                if saved_templates:
                    last_id, last_x, last_y, last_path = saved_templates.pop()
                    
                    # Delete files
                    if last_path.exists():
                        last_path.unlink()
                    json_path = last_path.with_suffix('.json')
                    if json_path.exists():
                        json_path.unlink()
                    
                    template_count -= 1
                    print(f"  ↶ Undid template {last_id}")
                    
                    # Redraw without last marker
                    display_img = current_img.copy()
                    info_text = f"Frame {frame_idx+1}/{len(frame_list)} | {frame_path.name} | Templates: {template_count}/{max_templates}"
                    cv2.putText(display_img, info_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_img, info_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                else:
                    print("  Nothing to undo")
            
            elif key == ord('q'):  # Quit
                print(f"\nQuitting. Total templates: {template_count}")
                cv2.destroyAllWindows()
                return template_count
        
        frame_idx += 1
    
    cv2.destroyAllWindows()
    
    print(f"\n{'='*70}")
    print(f"COMPLETED!")
    print(f"{'='*70}")
    print(f"Total templates created: {template_count}")
    print(f"Saved to: {output_path}")
    
    if template_count >= max_templates:
        print(f"\n✓ Target reached: {max_templates} templates!")
    else:
        print(f"\nCreated {template_count}/{max_templates} templates")
        print(f"Run again to add more")
    
    print(f"\n{'='*70}")
    print(f"Next steps:")
    print(f"  Test: python detect_placement.py detect <frame> --templates {output_path}")
    print(f"  Batch: python create_placement_dataset.py <data_dir> --method template --templates {output_path}")
    print(f"{'='*70}")
    
    return template_count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fast clock template extractor - just click on clocks!"
    )
    parser.add_argument("frames_dir", help="Directory containing frames (searches recursively)")
    parser.add_argument("--output", "-o", default="clock_templates",
                       help="Output directory (default: clock_templates)")
    parser.add_argument("--size", "-s", type=int, default=35,
                       help="Template size in pixels (default: 35)")
    parser.add_argument("--max", "-m", type=int, default=30,
                       help="Maximum templates to create (default: 30)")
    
    args = parser.parse_args()
    
    fast_extract_templates(
        frames_dir=args.frames_dir,
        output_dir=args.output,
        template_size=args.size,
        max_templates=args.max
    )


if __name__ == "__main__":
    main()

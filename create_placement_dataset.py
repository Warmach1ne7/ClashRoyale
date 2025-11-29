"""
Complete pipeline for troop placement detection and dataset creation.
Detects blue clock icons and outputs training data in format: troop, x, y, arena, frame
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from detect_placement import PlacementDetector
from cr_element import crop_arena

def extract_arena_and_frame_info(image_path: Path) -> Tuple[str, int]:
    """
    Extract arena and frame number from path structure.
    Expected: .../arena_XX/game_YY/images/frame_ZZZZ.png
    
    Args:
        image_path: Path to the frame image
    
    Returns:
        (arena, frame_number) tuple
    """
    parts = image_path.parts
    
    # Find arena
    arena = "unknown"
    for part in parts:
        if part.startswith("arena_"):
            arena = part.replace("arena_", "")
            break
    
    # Extract frame number
    frame_num = 0
    stem = image_path.stem  # e.g., "frame_0123"
    if "frame_" in stem or stem.isdigit():
        try:
            frame_num = int(stem.split("_")[-1])
        except (ValueError, IndexError):
            frame_num = 0
    
    return arena, frame_num


def detect_blue_clock_placements(image_path: str, 
                                 detector: PlacementDetector,
                                 min_radius: int = 10,
                                 max_radius: int = 30) -> List[Tuple[int, int, float, str]]:
    """
    Detect clock placements using template matching.
    No color filtering - you'll verify with card placement data.
    
    Args:
        image_path: Path to game frame
        detector: PlacementDetector instance with loaded templates
        min_radius: Minimum clock radius in pixels (unused, kept for compatibility)
        max_radius: Maximum clock radius in pixels (unused, kept for compatibility)
    
    Returns:
        List of (x, y, confidence, template_id) tuples (all detected clocks)
    """
    raw_img = cv2.imread(image_path)
    img = crop_arena(raw_img)
    # Run template matching with all templates
    detections = detector.detect_placements(
        img, 
        use_all_templates=True,
        return_template_id=True
    )
    
    return detections


def process_game_directory(game_dir: Path,
                          detector: Optional[PlacementDetector] = None,
                          detection_method: str = "template",
                          troop_name: str = "unknown") -> pd.DataFrame:
    """
    Process all frames in a game directory and detect placements.
    
    Args:
        game_dir: Path to game directory with images/
        detector: PlacementDetector with loaded templates (required for template method)
        detection_method: "template" (default) or "color"
        troop_name: Default troop name (can be refined later)
    
    Returns:
        DataFrame with columns: troop, x, y, arena, frame, game_id, template_id
    """
    images_dir = game_dir / "images"
    if not images_dir.exists():
        print(f"No images directory found in {game_dir}")
        return pd.DataFrame(columns=['troop', 'x', 'y', 'arena', 'frame', 'game_id', 'template_id'])
    
    # Extract game_id from directory name (UUID or game_XX format)
    game_id = game_dir.name
    
    # Get arena info from path
    arena, _ = extract_arena_and_frame_info(images_dir / "dummy.png")
    
    all_placements = []
    
    image_files = sorted(images_dir.glob("*.png"))
    print(f"Processing {len(image_files)} frames from {game_dir.name}...")
    
    for img_path in image_files:
        # Extract frame number
        _, frame_num = extract_arena_and_frame_info(img_path)
        
        # Detect placements
        if detection_method == "template" and detector:
            detections = detect_blue_clock_placements(str(img_path), detector)
        else:
            print(f"Warning: detection_method={detection_method} not fully supported. Use 'template'.")
            continue
        
        # Add to results (now detections are 4-tuples: x, y, confidence, template_id)
        for x, y, confidence, template_id in detections:
            all_placements.append({
                'troop': troop_name,
                'x': x,
                'y': y,
                'arena': arena,
                'frame': frame_num,
                'game_id': game_id,
                'template_id': template_id,
                'confidence': confidence
            })
    
    df = pd.DataFrame(all_placements)
    
    if len(df) > 0:
        print(f"  Found {len(df)} placements across {df['frame'].nunique()} frames")
    else:
        print(f"  No placements detected")
    
    return df


def process_multiple_games(data_root: Path,
                          arenas: List[str] = None,
                          output_csv: str = "troop_placements.csv",
                          detection_method: str = "template",
                          template_dir: Optional[str] = None,
                          game_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Process multiple games across arenas and create unified dataset.
    
    Args:
        data_root: Root data directory
        arenas: List of arena names (e.g., ["arena_01", "arena_02"])
        output_csv: Output CSV filename
        detection_method: "template" (default)
        template_dir: Directory with clock templates (required for template method)
        game_filter: If specified, only process games matching this UUID or pattern
    
    Returns:
        Combined DataFrame with game_id and template_id columns
    """
    # Setup detector if using template method
    detector = None
    if detection_method == "template":
        if not template_dir:
            raise ValueError("template_dir required for template detection method")
        detector = PlacementDetector(threshold=0.65)
        detector.load_multiple_templates(template_dir)
    
    # If no arenas specified, find all arena directories
    if arenas is None:
        arenas = [d.name for d in data_root.iterdir() 
                 if d.is_dir() and d.name.startswith("arena_")]
        arenas = sorted(arenas)
    
    print(f"\n{'='*70}")
    print(f"TROOP PLACEMENT DETECTION PIPELINE")
    print(f"{'='*70}")
    print(f"Data root: {data_root}")
    print(f"Detection method: {detection_method}")
    print(f"Arenas to process: {len(arenas)}")
    print(f"{'='*70}\n")
    
    all_data = []
    
    for arena_name in arenas:
        arena_dir = data_root / arena_name
        if not arena_dir.exists():
            continue
        
        print(f"\n[{arena_name}]")
        
        # Find all game directories - handle both "game_XX" and UUID formats
        game_dirs = sorted([d for d in arena_dir.iterdir() 
                          if d.is_dir() and (d.name.startswith("game_") or 
                                            len(d.name) == 36 or  # UUID length
                                            (d / "images").exists())])  # Has images folder
        
        # Apply game filter if specified
        if game_filter:
            game_dirs = [d for d in game_dirs if game_filter in d.name]
        
        if not game_dirs:
            msg = f"  No game directories found in {arena_name}"
            if game_filter:
                msg += f" matching '{game_filter}'"
            print(msg)
            continue
        
        print(f"  Found {len(game_dirs)} game directories")
        
        for game_dir in game_dirs:
            print(f"  {game_dir.name}:")
            
            # Process this game
            df = process_game_directory(
                game_dir,
                detector=detector,
                detection_method=detection_method
            )
            
            if len(df) > 0:
                all_data.append(df)
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by arena, game_id, and frame
        combined_df = combined_df.sort_values(['arena', 'game_id', 'frame'])
        
        # Save to CSV with game_id and template_id for debugging
        output_df = combined_df[['troop', 'x', 'y', 'arena', 'frame', 'game_id', 'template_id']]
        output_df.to_csv(output_csv, index=False)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Total placements detected: {len(combined_df)}")
        print(f"Unique games processed: {combined_df['game_id'].nunique()}")
        print(f"Arenas covered: {combined_df['arena'].nunique()}")
        print(f"Frames with placements: {combined_df['frame'].nunique()}")
        print(f"Templates used: {combined_df['template_id'].nunique()}")
        print(f"\nOutput saved to: {output_csv}")
        print(f"Columns: troop, x, y, arena, frame, game_id, template_id")
        print(f"{'='*70}\n")
        
        # Show sample
        print("Sample data:")
        print(output_df.head(10))
        
        return combined_df
    else:
        print("\n⚠ No placements detected in any games!")
        return pd.DataFrame(columns=['troop', 'x', 'y', 'arena', 'frame', 'game_id', 'template_id'])


def temporal_filtering(df: pd.DataFrame, 
                       min_consecutive_frames: int = 3) -> pd.DataFrame:
    """
    Filter out spurious detections by requiring placements to appear
    in multiple consecutive frames (since clock lasts ~10 frames at 10fps).
    
    Args:
        df: DataFrame with columns troop, x, y, arena, frame, game_id, template_id, confidence
        min_consecutive_frames: Minimum frames required (default: 3)
    
    Returns:
        Filtered DataFrame
    """
    if len(df) == 0:
        return df
    
    print(f"\nApplying temporal filtering (min {min_consecutive_frames} consecutive frames)...")
    
    # Group by arena and game_id
    filtered_groups = []
    
    for (arena, game_id) in df[['arena', 'game_id']].drop_duplicates().values:
        game_df = df[(df['arena'] == arena) & (df['game_id'] == game_id)].sort_values('frame')
        
        # Track placement clusters across frames
        valid_placements = []
        
        for idx, row in game_df.iterrows():
            x, y, frame = row['x'], row['y'], row['frame']
            
            # Check if this placement appears in nearby frames
            nearby_frames = game_df[
                (game_df['frame'] >= frame - 2) & 
                (game_df['frame'] <= frame + 2) &
                (game_df['frame'] != frame)
            ]
            
            # Count how many nearby frames have placements at similar location
            similar_count = 0
            for _, nearby_row in nearby_frames.iterrows():
                distance = np.sqrt((x - nearby_row['x'])**2 + (y - nearby_row['y'])**2)
                if distance < 30:  # Within 30 pixels
                    similar_count += 1
            
            # Keep if appears in enough frames
            if similar_count >= min_consecutive_frames - 1:
                valid_placements.append(row)
        
        if valid_placements:
            filtered_groups.append(pd.DataFrame(valid_placements))
    
    if filtered_groups:
        filtered_df = pd.concat(filtered_groups, ignore_index=True)
        print(f"  Before: {len(df)} placements")
        print(f"  After: {len(filtered_df)} placements")
        print(f"  Removed: {len(df) - len(filtered_df)} spurious detections")
        return filtered_df
    else:
        return pd.DataFrame(columns=df.columns)


def visualize_detections_on_frame(frame_path: str, 
                                 detections: List[Tuple],
                                 output_path: str):
    """Create visualization of detections on a frame."""
    raw_img = cv2.imread(frame_path)
    img = crop_arena(img)
    if img is None:
        return
    
    for det in detections:
        x, y = det[0], det[1]
        conf = det[2] if len(det) > 2 else 1.0
        template_id = det[3] if len(det) > 3 else 'unknown'
        
        # Draw circle at placement
        cv2.circle(img, (x, y), 15, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        
        # Draw crosshair
        cv2.line(img, (x-20, y), (x+20, y), (0, 255, 0), 2)
        cv2.line(img, (x, y-20), (x, y+20), (0, 255, 0), 2)
        
        # Add confidence and template info
        text = f"{conf:.2f}"
        if template_id != 'unknown':
            text += f" ({template_id})"
        cv2.putText(img, text, (x+20, y-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, img)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Detect troop placements and create training dataset"
    )
    parser.add_argument("data_dir", help="Root data directory (contains arena_XX folders)")
    parser.add_argument("--output", "-o", default="troop_placements.csv",
                       help="Output CSV file (default: troop_placements.csv)")
    parser.add_argument("--method", "-m", choices=["template"], 
                       default="template",
                       help="Detection method (default: template)")
    parser.add_argument("--templates", "-t", type=str, required=True,
                       help="Template directory (required)")
    parser.add_argument("--arenas", "-a", nargs="+",
                       help="Specific arenas to process (e.g., arena_01 arena_02)")
    parser.add_argument("--game", "-g", type=str, default=None,
                       help="Process only specific game (UUID or substring match)")
    parser.add_argument("--filter", "-f", action="store_true",
                       help="Apply temporal filtering to remove spurious detections")
    parser.add_argument("--visualize", "-v", type=str,
                       help="Visualize detections on a sample frame")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_dir)
    
    if not data_root.exists():
        print(f"Error: Data directory not found: {data_root}")
        return
    
    # Process all games
    df = process_multiple_games(
        data_root=data_root,
        arenas=args.arenas,
        output_csv=args.output,
        detection_method=args.method,
        template_dir=args.templates,
        game_filter=args.game
    )
    
    # Apply temporal filtering if requested
    if args.filter and len(df) > 0:
        df = temporal_filtering(df)
        # Re-save filtered results
        output_df = df[['troop', 'x', 'y', 'arena', 'frame', 'game_id', 'template_id']]
        output_df.to_csv(args.output, index=False)
        print(f"\nFiltered data saved to: {args.output}")
    
    # Visualize if requested
    if args.visualize and len(df) > 0:
        sample_frame = df.iloc[0]
        # Use game_id instead of hardcoded game_01
        frame_path = data_root / f"arena_{sample_frame['arena']}" / sample_frame['game_id'] / "images" / f"frame_{sample_frame['frame']:04d}.png"
        
        if frame_path.exists():
            # Get detections for this frame
            detector = PlacementDetector(threshold=0.65)
            detector.load_multiple_templates(args.templates)
            detections = detector.detect_placements(str(frame_path), return_template_id=True)
            
            visualize_detections_on_frame(str(frame_path), detections, args.visualize)
            print(f"Visualization saved to: {args.visualize}")


if __name__ == "__main__":
    main()

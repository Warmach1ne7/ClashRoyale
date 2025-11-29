"""
Update training parquet file with placement coordinates from detection CSV.

For each card placement in training data:
- Search detection CSV for matching placement in 5-frame window
- If found: use detected x,y coordinates
- If not found: use -1,-1 (no placement detected)
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional
import argparse


def load_detection_csv(csv_path: str) -> pd.DataFrame:
    """Load detection CSV and prepare for fast lookup"""
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required = ['x', 'y', 'arena', 'frame', 'game_id']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Detection CSV missing columns: {missing}")
    
    print(f"Loaded {len(df)} detections from {csv_path}")
    print(f"  Unique games: {df['game_id'].nunique()}")
    print(f"  Unique arenas: {df['arena'].nunique()}")
    print(f"  Frame range: {df['frame'].min()} - {df['frame'].max()}")
    
    return df


def find_placement_in_window(detections: pd.DataFrame, 
                             arena: str, 
                             game_id: str, 
                             frame: int,
                             window_size: int = 5) -> tuple:
    """
    Search for placement detection in frame window.
    
    Args:
        detections: DataFrame with detection results
        arena: Arena number (as string, e.g., "31")
        game_id: Game UUID or ID
        frame: Center frame to search around
        window_size: Number of frames to search (frame to frame+window_size)
    
    Returns:
        (x, y) tuple if found, (-1, -1) if not found
    """
    # Search in frame window: [frame, frame+1, ..., frame+window_size-1]
    frame_start = frame
    frame_end = frame + window_size
    
    # Convert arena to int for comparison with detection CSV
    arena_int = int(arena) if arena.isdigit() else int(arena.replace('arena_', ''))
    
    # Filter detections for this game, arena, and frame window
    mask = (
        (detections['arena'] == arena_int) &
        (detections['game_id'] == game_id) &
        (detections['frame'] >= frame_start) &
        (detections['frame'] < frame_end)
    )
    
    matches = detections[mask]
    
    if len(matches) == 0:
        return (-1, -1)
    
    # If multiple matches, take the closest frame
    matches = matches.copy()
    matches['frame_dist'] = abs(matches['frame'] - frame)
    closest = matches.sort_values('frame_dist').iloc[0]
    
    return (int(closest['x']), int(closest['y']))


def update_training_parquet(parquet_path: str,
                            detection_csv: str,
                            output_path: str,
                            window_size: int = 5):
    """
    Update training parquet with detected placement coordinates.
    
    Args:
        parquet_path: Path to training parquet file
        detection_csv: Path to detection CSV from create_placement_dataset.py
        output_path: Path to save updated parquet
        window_size: Frame window size for searching detections
    """
    print("\n" + "="*70)
    print("UPDATE TRAINING PARQUET WITH DETECTIONS")
    print("="*70)
    
    # Load detection results
    print("\n[1/4] Loading detection CSV...")
    detections = load_detection_csv(detection_csv)
    
    # Load training parquet (exclude png_bytes to save memory)
    print("\n[2/4] Loading training parquet...")
    # Read only necessary columns to avoid loading image bytes
    columns_to_read = ['card', 'x', 'y', 'arena', 'replay', 'frame']
    train_df = pd.read_parquet(parquet_path, columns=columns_to_read)
    
    print(f"Training data: {len(train_df)} rows")
    print(f"Columns: {list(train_df.columns)}")
    
    # Verify required columns
    required = ['card', 'arena', 'replay', 'frame']
    missing = [col for col in required if col not in train_df.columns]
    if missing:
        raise ValueError(f"Training parquet missing columns: {missing}")
    
    # Add x,y columns if they don't exist
    if 'x' not in train_df.columns:
        train_df['x'] = -1
    if 'y' not in train_df.columns:
        train_df['y'] = -1
    
    # Process each row
    print("\n[3/4] Matching detections to training data...")
    matched_count = 0
    unmatched_count = 0
    
    for idx, row in train_df.iterrows():
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{len(train_df)} rows ({matched_count} matched, {unmatched_count} unmatched)")
        
        # Convert arena format: "arena_31" -> "31"
        arena_str = str(row['arena'])
        arena = arena_str.replace('arena_', '') if 'arena_' in arena_str else arena_str
        game_id = str(row['replay'])
        frame = int(row['frame'])
        
        # Find placement in detection window
        x, y = find_placement_in_window(detections, arena, game_id, frame, window_size)
        
        train_df.at[idx, 'x'] = x
        train_df.at[idx, 'y'] = y
        
        if x != -1 and y != -1:
            matched_count += 1
        else:
            unmatched_count += 1
    
    print(f"\n  Final: {matched_count} matched, {unmatched_count} unmatched")
    
    # Save updated parquet (need to merge back with original to preserve png_bytes)
    print("\n[4/4] Saving updated parquet...")
    
    # Read original parquet and update only x,y columns
    print("  Loading original parquet to preserve png_bytes...")
    original_df = pd.read_parquet(parquet_path)
    original_df['x'] = train_df['x']
    original_df['y'] = train_df['y']
    
    print(f"  Writing to {output_path}...")
    original_df.to_parquet(output_path, index=False)
    
    print(f"\n✓ Saved to: {output_path}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total rows: {len(train_df)}")
    print(f"Matched placements: {matched_count} ({matched_count/len(train_df)*100:.1f}%)")
    print(f"Unmatched placements: {unmatched_count} ({unmatched_count/len(train_df)*100:.1f}%)")
    print(f"\nCoordinate statistics:")
    print(f"  X range: {train_df[train_df['x'] != -1]['x'].min():.0f} - {train_df[train_df['x'] != -1]['x'].max():.0f}")
    print(f"  Y range: {train_df[train_df['y'] != -1]['y'].min():.0f} - {train_df[train_df['y'] != -1]['y'].max():.0f}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Update training parquet with placement coordinates from detection CSV"
    )
    parser.add_argument("parquet_path", 
                       help="Path to training parquet file (download from HuggingFace first)")
    parser.add_argument("detection_csv",
                       help="Path to detection CSV from create_placement_dataset.py")
    parser.add_argument("--output", "-o", default="train_with_coords.parquet",
                       help="Output parquet file (default: train_with_coords.parquet)")
    parser.add_argument("--window", "-w", type=int, default=5,
                       help="Frame window size for searching detections (default: 5)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.parquet_path).exists():
        print(f"Error: Parquet file not found: {args.parquet_path}")
        print("\nTo download from HuggingFace:")
        print("  from datasets import load_dataset")
        print("  ds = load_dataset('your-username/clash-royale')")
        print("  ds['train'].to_parquet('train.parquet')")
        return
    
    if not Path(args.detection_csv).exists():
        print(f"Error: Detection CSV not found: {args.detection_csv}")
        return
    
    # Run update
    update_training_parquet(
        args.parquet_path,
        args.detection_csv,
        args.output,
        args.window
    )


if __name__ == "__main__":
    main()

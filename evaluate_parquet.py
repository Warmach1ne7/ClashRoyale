"""
Evaluate parquet file to check detection coverage and missing placements.

Reports statistics on:
- How many placements have coordinates vs -1,-1 (not detected)
- Breakdown by arena, replay, and card type
- Detection rate percentages
"""

import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict


def evaluate_parquet(parquet_path: str, 
                     arenas: list = None, 
                     replays: list = None,
                     verbose: bool = False):
    """
    Evaluate detection coverage in parquet file.
    
    Args:
        parquet_path: Path to parquet file with x,y coordinates
        arenas: List of arenas to filter (e.g., ['arena_31', 'arena_30'])
        replays: List of replay IDs to filter (e.g., ['5eacd6a6-1169-46df...'])
        verbose: Show detailed breakdown by card type
    """
    print("\n" + "="*70)
    print("PARQUET DETECTION EVALUATION")
    print("="*70)
    
    # Load parquet (only needed columns)
    print("\nLoading parquet file...")
    columns = ['card', 'x', 'y', 'arena', 'replay', 'frame']
    df = pd.read_parquet(parquet_path, columns=columns)
    
    print(f"Total rows: {len(df)}")
    
    # Apply filters
    if arenas:
        print(f"Filtering arenas: {arenas}")
        df = df[df['arena'].isin(arenas)]
        print(f"  After filter: {len(df)} rows")
    
    if replays:
        print(f"Filtering replays: {len(replays)} replay(s)")
        df = df[df['replay'].isin(replays)]
        print(f"  After filter: {len(df)} rows")
    
    if len(df) == 0:
        print("\n✗ No data after filtering!")
        return
    
    # Calculate detection stats
    total = len(df)
    detected = len(df[(df['x'] != -1) & (df['y'] != -1)])
    missing = total - detected
    detection_rate = (detected / total * 100) if total > 0 else 0
    
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    print(f"Total placements:     {total:>8,}")
    print(f"Detected:             {detected:>8,}  ({detection_rate:.1f}%)")
    print(f"Missing (x=-1, y=-1): {missing:>8,}  ({100-detection_rate:.1f}%)")
    
    # Breakdown by arena
    print("\n" + "="*70)
    print("BREAKDOWN BY ARENA")
    print("="*70)
    print(f"{'Arena':<15} {'Total':>8} {'Detected':>10} {'Missing':>10} {'Rate':>8}")
    print("-"*70)
    
    for arena in sorted(df['arena'].unique()):
        arena_df = df[df['arena'] == arena]
        arena_total = len(arena_df)
        arena_detected = len(arena_df[(arena_df['x'] != -1) & (arena_df['y'] != -1)])
        arena_missing = arena_total - arena_detected
        arena_rate = (arena_detected / arena_total * 100) if arena_total > 0 else 0
        
        print(f"{arena:<15} {arena_total:>8,} {arena_detected:>10,} {arena_missing:>10,} {arena_rate:>7.1f}%")
    
    # Breakdown by replay
    print("\n" + "="*70)
    print("BREAKDOWN BY REPLAY (Top 20 by count)")
    print("="*70)
    print(f"{'Replay ID':<40} {'Total':>8} {'Detected':>10} {'Missing':>10} {'Rate':>8}")
    print("-"*70)
    
    replay_stats = []
    for replay in df['replay'].unique():
        replay_df = df[df['replay'] == replay]
        replay_total = len(replay_df)
        replay_detected = len(replay_df[(replay_df['x'] != -1) & (replay_df['y'] != -1)])
        replay_missing = replay_total - replay_detected
        replay_rate = (replay_detected / replay_total * 100) if replay_total > 0 else 0
        replay_stats.append((replay, replay_total, replay_detected, replay_missing, replay_rate))
    
    # Sort by total count and show top 20
    replay_stats.sort(key=lambda x: x[1], reverse=True)
    for replay, total, detected, missing, rate in replay_stats[:20]:
        replay_short = replay[:37] + "..." if len(replay) > 40 else replay
        print(f"{replay_short:<40} {total:>8,} {detected:>10,} {missing:>10,} {rate:>7.1f}%")
    
    if len(replay_stats) > 20:
        print(f"... and {len(replay_stats) - 20} more replays")
    
    # Breakdown by card type (if verbose)
    if verbose:
        print("\n" + "="*70)
        print("BREAKDOWN BY CARD TYPE")
        print("="*70)
        print(f"{'Card':<20} {'Total':>8} {'Detected':>10} {'Missing':>10} {'Rate':>8}")
        print("-"*70)
        
        card_stats = []
        for card in sorted(df['card'].unique()):
            card_df = df[df['card'] == card]
            card_total = len(card_df)
            card_detected = len(card_df[(card_df['x'] != -1) & (card_df['y'] != -1)])
            card_missing = card_total - card_detected
            card_rate = (card_detected / card_total * 100) if card_total > 0 else 0
            card_stats.append((card, card_total, card_detected, card_missing, card_rate))
        
        # Sort by detection rate (ascending - worst first)
        card_stats.sort(key=lambda x: x[4])
        for card, total, detected, missing, rate in card_stats:
            print(f"{card:<20} {total:>8,} {detected:>10,} {missing:>10,} {rate:>7.1f}%")
    
    # Frame range stats
    if detected > 0:
        detected_df = df[(df['x'] != -1) & (df['y'] != -1)]
        print("\n" + "="*70)
        print("COORDINATE STATISTICS (Detected placements only)")
        print("="*70)
        print(f"X coordinate range: {detected_df['x'].min():.0f} - {detected_df['x'].max():.0f}")
        print(f"Y coordinate range: {detected_df['y'].min():.0f} - {detected_df['y'].max():.0f}")
        print(f"Frame range:        {detected_df['frame'].min():.0f} - {detected_df['frame'].max():.0f}")
    
    # Low detection replays
    low_threshold = 30.0  # Less than 30% detection
    low_detection_replays = [r for r in replay_stats if r[4] < low_threshold]
    
    if low_detection_replays:
        print("\n" + "="*70)
        print(f"REPLAYS WITH LOW DETECTION (<{low_threshold}%)")
        print("="*70)
        print(f"{'Replay ID':<40} {'Total':>8} {'Detected':>10} {'Missing':>10} {'Rate':>8}")
        print("-"*70)
        
        for replay, total, detected, missing, rate in sorted(low_detection_replays, key=lambda x: x[4]):
            replay_short = replay[:37] + "..." if len(replay) > 40 else replay
            print(f"{replay_short:<40} {total:>8,} {detected:>10,} {missing:>10,} {rate:>7.1f}%")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate detection coverage in parquet file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate entire parquet
  python evaluate_parquet.py train_with_coords.parquet
  
  # Evaluate specific arena
  python evaluate_parquet.py train_with_coords.parquet --arenas arena_31
  
  # Evaluate multiple arenas
  python evaluate_parquet.py train_with_coords.parquet --arenas arena_31 arena_30
  
  # Evaluate specific replays
  python evaluate_parquet.py train_with_coords.parquet --replays 5eacd6a6-1169-46df-b5e5-4b10798678fb
  
  # Show detailed card type breakdown
  python evaluate_parquet.py train_with_coords.parquet --verbose
        """
    )
    
    parser.add_argument("parquet_path",
                       help="Path to parquet file with x,y coordinates")
    parser.add_argument("--arenas", "-a", nargs="+",
                       help="Filter by specific arenas (e.g., arena_31 arena_30)")
    parser.add_argument("--replays", "-r", nargs="+",
                       help="Filter by specific replay IDs")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed breakdown by card type")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.parquet_path).exists():
        print(f"Error: Parquet file not found: {args.parquet_path}")
        return
    
    evaluate_parquet(
        args.parquet_path,
        arenas=args.arenas,
        replays=args.replays,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()

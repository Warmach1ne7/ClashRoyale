"""
Quick test script to verify the detection changes work correctly.
Tests game_id, template_id tracking, and opponent filtering.
"""

from pathlib import Path
from detect_placement import PlacementDetector
from create_placement_dataset import detect_blue_clock_placements, process_game_directory
import pandas as pd

def test_detection():
    print("Testing enhanced detection with game_id and template_id...")
    
    # Setup paths
    data_root = Path("../hf_subset")
    template_dir = "clock_templates"
    arena_dir = data_root / "arena_11"
    
    # Check if paths exist
    if not arena_dir.exists():
        print(f"Error: {arena_dir} not found")
        return
    
    # Find first game directory
    game_dirs = [d for d in arena_dir.iterdir() if d.is_dir() and (d / "images").exists()]
    if not game_dirs:
        print("No game directories found!")
        return
    
    game_dir = game_dirs[0]
    print(f"\nTesting with: {game_dir}")
    print(f"Game ID: {game_dir.name}")
    
    # Load detector
    print("\nLoading templates...")
    detector = PlacementDetector(threshold=0.65)
    detector.load_multiple_templates(template_dir)
    
    # Test single frame detection
    images_dir = game_dir / "images"
    test_frames = sorted(images_dir.glob("*.png"))[:5]
    
    print(f"\nTesting on {len(test_frames)} frames...")
    for frame_path in test_frames:
        detections = detect_blue_clock_placements(str(frame_path), detector)
        
        if detections:
            print(f"\nFrame: {frame_path.name}")
            print(f"  Detections: {len(detections)}")
            for x, y, conf, template_id in detections:
                print(f"    - ({x}, {y}) conf={conf:.3f} template={template_id}")
    
    # Test full game processing
    print("\n" + "="*70)
    print("Testing full game processing...")
    print("="*70)
    
    df = process_game_directory(
        game_dir,
        detector=detector,
        detection_method="template"
    )
    
    if len(df) > 0:
        print(f"\n✓ Successfully detected {len(df)} placements")
        print(f"✓ Columns: {list(df.columns)}")
        print(f"✓ Game ID: {df['game_id'].iloc[0]}")
        print(f"✓ Templates used: {df['template_id'].unique()}")
        print(f"\nSample data:")
        print(df.head(10))
        
        # Save test output
        output_file = "test_detections.csv"
        df[['troop', 'x', 'y', 'arena', 'frame', 'game_id', 'template_id']].to_csv(output_file, index=False)
        print(f"\n✓ Test output saved to: {output_file}")
    else:
        print("\n⚠ No detections found!")
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    test_detection()

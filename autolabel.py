"""
Generate YOLO labels for all images using the ROIs from rois.json.
Creates labels for king and princess towers across all arena/game folders.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Class mapping
CLASS_MAP = {
    'king': 0,
    'princess': 1
}

def roi_to_yolo_box(roi: List[float]) -> Tuple[float, float, float, float]:
    """
    Convert ROI [x1, y1, x2, y2] (normalized) to YOLO format.
    YOLO: [x_center, y_center, width, height] (all normalized)
    """
    x1, y1, x2, y2 = roi
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return x_center, y_center, width, height

def create_label_file(image_path: Path, rois: Dict, labels_dir: Path):
    """
    Create a YOLO label file for a given image using the ROI definitions.
    """
    label_path = labels_dir / f"{image_path.stem}.txt"
    
    boxes = []
    
    # King towers (2 total: top and bottom)
    for roi_name in ['king_top', 'king_bottom']:
        if roi_name in rois:
            roi = rois[roi_name]
            # Skip invalid/empty ROIs
            if roi[0] == roi[2] and roi[1] == roi[3]:
                continue
            x_c, y_c, w, h = roi_to_yolo_box(roi)
            boxes.append(f"{CLASS_MAP['king']} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    
    # Princess towers (4 total: top-left, top-right, bottom-left, bottom-right)
    for roi_name in ['princess_top_l', 'princess_top_r', 'princess_bot_l', 'princess_bot_r']:
        if roi_name in rois:
            roi = rois[roi_name]
            # Skip invalid/empty ROIs
            if roi[0] == roi[2] and roi[1] == roi[3]:
                continue
            x_c, y_c, w, h = roi_to_yolo_box(roi)
            boxes.append(f"{CLASS_MAP['princess']} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    
    # Write label file
    if boxes:
        with open(label_path, 'w') as f:
            f.write('\n'.join(boxes) + '\n')

def main():
    # Paths
    data_root = Path('/home/ostikar/MyProjects/CS541/ClashRoyale/data')
    rois_json = data_root / 'towers' / 'rois.json'
    
    # Load ROIs
    print(f"Loading ROIs from {rois_json}...")
    with open(rois_json, 'r') as f:
        rois = json.load(f)
    
    print(f"ROIs loaded: {list(rois.keys())}")
    
    # Find arena directories 01-10 only
    arena_dirs = []
    for i in range(1, 11):  # arenas 01 through 10
        arena_name = f'arena_{i:02d}'
        arena_path = data_root / arena_name
        if arena_path.exists():
            arena_dirs.append(arena_path)
    
    print(f"\nFound {len(arena_dirs)} arena directories (01-10)")
    
    total_images = 0
    total_labels = 0
    
    # Process each arena
    for arena_dir in arena_dirs:
        print(f"\nProcessing {arena_dir.name}...")
        
        # Find all game directories
        game_dirs = sorted([d for d in arena_dir.iterdir() if d.is_dir() and d.name.startswith('game_')])
        
        for game_dir in game_dirs:
            images_dir = game_dir / 'images'
            if not images_dir.exists():
                continue
            
            # Create labels directory
            labels_dir = game_dir / 'labels'
            labels_dir.mkdir(exist_ok=True)
            
            # Process all images
            image_files = sorted(images_dir.glob('*.png')) + sorted(images_dir.glob('*.jpg'))
            
            for img_path in image_files:
                create_label_file(img_path, rois, labels_dir)
                total_labels += 1
            
            total_images += len(image_files)
            
            if len(image_files) > 0:
                print(f"  {game_dir.name}: {len(image_files)} images labeled")
    
    print(f"\n{'='*60}")
    print(f"Total images found: {total_images}")
    print(f"Total label files created: {total_labels}")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Run split_yolo_dataset.py to create train/val/test splits")
    print(f"2. Update data.yaml to point to the split directories")
    print(f"3. Train with tower_train_improved.py")

if __name__ == '__main__':
    main()
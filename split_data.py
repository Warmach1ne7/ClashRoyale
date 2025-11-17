"""
Split the dataset into train/val/test sets and organize for YOLO training.
Creates a consolidated dataset structure with proper splits.
"""
import shutil
from pathlib import Path
import random
from typing import List, Tuple
import json

def collect_all_images(data_root: Path) -> List[Tuple[Path, Path]]:
    """
    Collect all (image, label) pairs from all arena/game folders.
    Returns list of (image_path, label_path) tuples.
    """
    pairs = []
    
    arena_dirs = []
    for i in range(1, 11):  # arenas 01 through 10
        arena_name = f'arena_{i:02d}'
        arena_path = data_root / arena_name
        if arena_path.exists():
            arena_dirs.append(arena_path)
    
    for arena_dir in arena_dirs:
        game_dirs = sorted([d for d in arena_dir.iterdir() if d.is_dir() and d.name.startswith('game_')])
        
        for game_dir in game_dirs:
            images_dir = game_dir / 'images'
            labels_dir = game_dir / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            # Include both PNG and JPG
            image_files = sorted(images_dir.glob('*.png')) + sorted(images_dir.glob('*.jpg'))
            for img_path in image_files:
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    pairs.append((img_path, label_path))
    
    return pairs

def create_split_dataset(
    pairs: List[Tuple[Path, Path]],
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split dataset and copy files to train/val/test directories.
    """
    # Clean output dir before creating new split
    if output_dir.exists():
        print(f"Removing existing dataset at {output_dir}...")
        shutil.rmtree(output_dir)
    
    random.seed(seed)
    random.shuffle(pairs)
    
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_pairs)} images")
    print(f"  Val:   {len(val_pairs)} images")
    print(f"  Test:  {len(test_pairs)} images")
    print(f"  Total: {len(pairs)} images")
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy files
    def copy_split(split_pairs: List[Tuple[Path, Path]], split_name: str):
        for img_src, label_src in split_pairs:
            img_dst = output_dir / split_name / 'images' / img_src.name
            label_dst = output_dir / split_name / 'labels' / label_src.name
            shutil.copy2(img_src, img_dst)
            shutil.copy2(label_src, label_dst)
    
    print("\nCopying files...")
    copy_split(train_pairs, 'train')
    copy_split(val_pairs, 'val')
    copy_split(test_pairs, 'test')
    
    print("✓ Dataset split complete!")
    
    # Save split metadata
    metadata = {
        'total': len(pairs),
        'train': len(train_pairs),
        'val': len(val_pairs),
        'test': len(test_pairs),
        'seed': seed,
        'ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}
    }
    
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    data_root = Path('/home/ostikar/MyProjects/CS541/ClashRoyale/data')
    output_dir = Path('/home/ostikar/MyProjects/CS541/ClashRoyale/data/yolo_dataset_health')
    
    print("Collecting all image-label pairs...")
    pairs = collect_all_images(data_root)
    print(f"Found {len(pairs)} valid image-label pairs")
    
    if len(pairs) == 0:
        print("\n⚠️  No image-label pairs found!")
        print("Make sure you've run autolabel_bars.py first")
        return
    
    create_split_dataset(
        pairs,
        output_dir,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
    
    print(f"\n{'='*60}")
    print(f"Dataset ready at: {output_dir}")
    print(f"\nYour data.yaml is already configured correctly:")
    print(f"  path: {output_dir}")
    print(f"  train: train/images")
    print(f"  val: val/images")
    print(f"  test: test/images")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
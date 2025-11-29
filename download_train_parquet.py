"""
Download training parquet file from HuggingFace dataset.
"""

import argparse
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not installed")
    print("Install with: pip install datasets")
    exit(1)


def download_train_parquet(dataset_name: str, output_path: str = "train.parquet"):
    """
    Download training split from HuggingFace dataset and save as parquet.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "username/clash-royale")
        output_path: Where to save the parquet file
    """
    print(f"\nDownloading dataset: {dataset_name}")
    print("="*70)
    
    try:
        # Load dataset
        print("\n[1/2] Loading from HuggingFace...")
        ds = load_dataset(dataset_name)
        
        print(f"  Dataset splits: {list(ds.keys())}")
        
        if 'train' not in ds:
            print(f"\nError: No 'train' split found in dataset")
            print(f"Available splits: {list(ds.keys())}")
            return
        
        train_ds = ds['train']
        print(f"  Training rows: {len(train_ds)}")
        print(f"  Columns: {train_ds.column_names}")
        
        # Save as parquet
        print(f"\n[2/2] Saving to {output_path}...")
        train_ds.to_parquet(output_path)
        
        file_size = Path(output_path).stat().st_size / (1024*1024)
        print(f"\n✓ Downloaded successfully!")
        print(f"  File: {output_path}")
        print(f"  Size: {file_size:.2f} MB")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nMake sure:")
        print("  1. Dataset name is correct (format: 'username/dataset-name')")
        print("  2. You have access to the dataset (public or you're logged in)")
        print("  3. You're logged in if needed: huggingface-cli login")


def main():
    parser = argparse.ArgumentParser(
        description="Download training parquet from HuggingFace dataset"
    )
    parser.add_argument("dataset", default="chrisrca/clash-royale-tv-replays", 
                       help="HuggingFace dataset name (e.g., 'username/clash-royale')")
    parser.add_argument("--output", "-o", default="train.parquet",
                       help="Output parquet file (default: train.parquet)")
    
    args = parser.parse_args()
    
    download_train_parquet(args.dataset, args.output)


if __name__ == "__main__":
    main()

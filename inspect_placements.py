import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
import random

# Path to your parquet file and image root
data_path = "training.parquet"  # Update if needed
image_root = "/home/ostikar/MyProjects/CS541/ClashRoyale/hf_subset"
df = pd.read_parquet(data_path)

print(f"Loaded parquet: {data_path}")
print(f"Total rows: {len(df)}")

# Drop rows with missing x/y/replay
samples = df.dropna(subset=['x', 'y', 'replay'])
print(f"Rows with x, y, replay: {len(samples)}")

# Randomly select 100 samples
samples_100 = samples.sample(n=min(100, len(samples)), random_state=42)

# --- PART 1: Overlay placements on images ---
output_dir = "placement_overlay_samples"
os.makedirs(output_dir, exist_ok=True)

for idx, row in samples_100.iterrows():
    replay = row['replay']
    arena = str(row['arena'])
    x, y = int(row['x']), int(row['y'])
    frame = int(row['frame']) if 'frame' in row and not pd.isnull(row['frame']) else None
    print(f"Processing replay: {replay}, frame: {frame}, x: {x}, y: {y}")
    img_dir = os.path.join(image_root, arena, str(replay), "images")
    if not os.path.isdir(img_dir):
        print(f"Image directory does not exist: {img_dir}")
        continue
    if frame is None:
        print(f"Frame is None for replay {replay}")
        continue
    img_filename = f"frame_{frame:06d}.png"
    img_path = os.path.join(img_dir, img_filename)
    if not os.path.isfile(img_path):
        # Try jpg or jpeg fallback
        img_filename_jpg = f"frame_{frame:06d}.jpg"
        img_filename_jpeg = f"frame_{frame:06d}.jpeg"
        if os.path.isfile(os.path.join(img_dir, img_filename_jpg)):
            img_path = os.path.join(img_dir, img_filename_jpg)
        elif os.path.isfile(os.path.join(img_dir, img_filename_jpeg)):
            img_path = os.path.join(img_dir, img_filename_jpeg)
        else:
            print(f"Image not found for replay {replay}, frame {frame} (tried {img_filename}, {img_filename_jpg}, {img_filename_jpeg}) in {img_dir}")
            continue
    print(f"Found image: {img_path}")
    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        r = 8
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0), outline=(0,0,0))
        # Save as placement_overlay_samples/arena/replay/frame_....jpg
        overlay_dir = os.path.join(output_dir, arena, str(replay))
        os.makedirs(overlay_dir, exist_ok=True)
        save_path = os.path.join(overlay_dir, f"frame_{frame:06d}_{x}_{y}.jpg")
        img.save(save_path)
        print(f"Saved overlay: {save_path}")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Count how many overlays were actually saved
num_saved = len([f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
if num_saved == 0:
    print(f"No overlays saved in {output_dir}/. Check if images exist.")
else:
    print(f"Saved {num_saved} overlays to {output_dir}/")

# --- PART 2: Heatmap of all placements ---
# Get all x, y for arena_31
all_x = samples['x'].astype(int)
all_y = samples['y'].astype(int)

# Assume image size from a sample image
sample_img_path = None
for replay in samples['replay'].unique():
    img_dir = os.path.join(image_root, str(replay), "images")
    if os.path.isdir(img_dir):
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if img_files:
            sample_img_path = os.path.join(img_dir, img_files[0])
            break
if sample_img_path:
    img = Image.open(sample_img_path)
    w, h = img.size
else:
    w, h = 640, 360  # fallback

heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=[w, h], range=[[0, w], [0, h]])
plt.figure(figsize=(10,6))
plt.imshow(heatmap.T, origin='lower', cmap='hot', alpha=0.7)
plt.title('Placement Heatmap for arena_31')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Count')
plt.savefig('arena_31_placement_heatmap.png', dpi=200)
plt.show()
print("Saved heatmap as arena_31_placement_heatmap.png")

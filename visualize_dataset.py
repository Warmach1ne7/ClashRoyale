import cv2
from pathlib import Path
import random

# Paths
DATA_DIR = Path("/home/ostikar/MyProjects/CS541/ClashRoyale/data/arena_05/game_2")
IMG_DIR = DATA_DIR / "images"
LBL_DIR = DATA_DIR / "labels"
OUTPUT_DIR = DATA_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# Class names and colors
CLASS_NAMES = {0: "king_tower", 1: "princess_tower"}
CLASS_COLORS = {
    0: (0, 255, 255),     # Yellow for king towers
    1: (255, 0, 255)      # Magenta for princess towers
}

def yolo_to_bbox(yolo_coords, img_w, img_h):
    """Convert YOLO format (cx, cy, w, h) to (x1, y1, x2, y2)"""
    cx, cy, w, h = yolo_coords
    x1 = int((cx - w/2) * img_w)
    y1 = int((cy - h/2) * img_h)
    x2 = int((cx + w/2) * img_w)
    y2 = int((cy + h/2) * img_h)
    return x1, y1, x2, y2

def visualize_image(img_path, label_path, save_path):
    """Visualize single image with bounding boxes"""
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read image: {img_path}")
        return False
    
    h, w = img.shape[:2]
    
    # Read labels
    if not label_path.exists():
        print(f"No label file for {img_path.name}")
        cv2.imwrite(str(save_path), img)
        return True
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Draw each bounding box
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        
        # Convert to pixel coordinates
        x1, y1, x2, y2 = yolo_to_bbox([cx, cy, bw, bh], w, h)
        
        # Draw bounding box
        color = CLASS_COLORS.get(class_id, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = CLASS_NAMES.get(class_id, f"class_{class_id}")
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Background for text
        cv2.rectangle(img, (x1, y1 - label_size[1] - 4), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Text
        cv2.putText(img, label, (x1, y1 - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Save
    cv2.imwrite(str(save_path), img)
    return True

def main(num_samples=None, random_sample=True):
    """
    Visualize dataset
    Args:
        num_samples: Number of images to visualize (None = all)
        random_sample: If True, randomly sample images
    """
    # Get all images
    image_files = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {IMG_DIR}")
        return
    
    # Sample if requested
    if num_samples and num_samples < len(image_files):
        if random_sample:
            image_files = random.sample(image_files, num_samples)
        else:
            image_files = image_files[:num_samples]
    
    print(f"Visualizing {len(image_files)} images...")
    print(f"Saving to: {OUTPUT_DIR}")
    
    success_count = 0
    for img_path in image_files:
        label_path = LBL_DIR / (img_path.stem + ".txt")
        save_path = OUTPUT_DIR / f"viz_{img_path.name}"
        
        if visualize_image(img_path, label_path, save_path):
            success_count += 1
            if success_count % 10 == 0:
                print(f"Processed {success_count}/{len(image_files)}")
    
    print(f"\n✅ Successfully visualized {success_count} images")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Print legend
    print("\n🎨 Color Legend:")
    print("   Yellow (0, 255, 255) = King Tower")
    print("   Magenta (255, 0, 255) = Princess Tower")

if __name__ == "__main__":
    # Visualize all images
    main(num_samples=10, random_sample=True)
    
    # Or visualize just 10 random samples:
    # main(num_samples=10, random_sample=True)
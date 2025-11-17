from ultralytics import YOLO
from pathlib import Path

# Load the trained model
MODEL_PATH = "runs/detect/tower_detection/weights/best.pt"
model = YOLO(MODEL_PATH)

# Path to test images
TEST_SOURCE = "/home/ostikar/MyProjects/CS541/ClashRoyale/data/arena_01/game_01"

# Run inference and save annotated images
results = model.predict(
    source=TEST_SOURCE,
    conf=0.25,  # Confidence threshold
    iou=0.45,   # NMS IOU threshold
    imgsz=640,
    save=True,  # Save annotated images with bounding boxes
    save_txt=True,  # Save detection labels
    show_labels=True,  # Show class names on boxes
    show_conf=True,  # Show confidence scores on boxes
    device=0,  # Use GPU
    project='runs/detect',
    name='inference_results',
    exist_ok=True,
    line_width=2,  # Bounding box thickness
    show_boxes=True  # Show bounding boxes
)

print(f"\nInference complete!")
print(f"Annotated images saved to: runs/detect/inference_results")
print(f"Total images processed: {len(results)}")

# Print summary of detections
for i, r in enumerate(results):
    img_name = Path(r.path).name
    num_detections = len(r.boxes)
    print(f"{img_name}: {num_detections} towers detected")
from ultralytics import YOLO
import wandb

wandb.init(project="clash-royale", name="towers_bars_finetune_v1")

# Load your best tower detection weights
model = YOLO('runs/detect/tower_detection4/weights/best.pt')

# Fine-tune to also detect health bars
results = model.train(
    data='data.yaml',
    epochs=50,  # Fewer epochs since starting from trained weights
    imgsz=640,
    batch=64,  # Increased batch size
    name='towers_bars_finetune',
    project='runs/detect',
    patience=12,  # Early stopping
    save=True,
    device=0,
    workers=8,  # More workers for larger dataset
    cache='disk',  # More deterministic than 'ram'
    verbose=True,
    # Augmentation settings
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    degrees=0.0,  # No rotation (UI is always upright)
    translate=0.05,  # Slight translation
    scale=0.3,  # Some scale variation
    fliplr=0.0,  # No horizontal flip (asymmetric game)
    flipud=0.0,  # No vertical flip
    mosaic=0.8,  # Reduced mosaic for cleaner training
    mixup=0.0,
    copy_paste=0.0,
)

# Validate the model
metrics = model.val()

# Test on held-out test set
test_metrics = model.val(split='test')

print(f"\n{'='*60}")
print(f"Fine-tuning complete!")
print(f"Best model: {model.trainer.best}")
print(f"\nValidation Metrics:")
print(f"  mAP50: {metrics.box.map50:.4f}")
print(f"  mAP50-95: {metrics.box.map:.4f}")
print(f"\nTest Metrics:")
print(f"  mAP50: {test_metrics.box.map50:.4f}")
print(f"  mAP50-95: {test_metrics.box.map:.4f}")

# Per-class metrics
class_names = ['king', 'princess', 'unused_2', 'unused_3', 'health_bar']
print(f"\nPer-class mAP50:")
for i, name in enumerate(class_names):
    if i < len(test_metrics.box.maps) and test_metrics.box.maps[i] > 0:
        print(f"  {name}: {test_metrics.box.maps[i]:.4f}")

print(f"{'='*60}")

wandb.finish()
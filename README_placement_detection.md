# Troop Placement Detection using Template Matching

This guide explains how to detect troop placement locations in Clash Royale using the placement clock icon.

## Overview

When a troop is placed in Clash Royale, a clock icon briefly appears at the placement location. We can use **template matching** to detect these clocks and pinpoint where troops were placed.

## Method: Template Matching

Template matching works by:
1. Creating a template (reference image) of the clock icon
2. Sliding this template across the game frame
3. Computing similarity scores at each position
4. Identifying locations with high similarity as detections

## Step-by-Step Usage

### Step 1: Create Clock Template

First, you need to create a template of the placement clock:

```bash
# Interactive tool - click and drag to select the clock region
python detect_placement.py create_template /path/to/frame/with/clock.png
```

This will:
- Open the image in a window
- Let you select the clock region by clicking and dragging
- Press 's' to save the template as `clock_template.png`

**Tips for selecting the template:**
- Choose a clear, high-contrast clock
- Include only the clock icon (not surrounding area)
- The clock should be from a typical placement (not scaled/rotated)

### Step 2: Detect Placements in Single Image

```bash
python detect_placement.py detect /path/to/test/frame.png clock_template.png
```

This will:
- Load the template
- Search for matching regions in the image
- Print detected positions and confidence scores
- Save a visualization with markers

### Step 3: Batch Detection Across Multiple Frames

```bash
python detect_placement.py batch /path/to/frames/directory clock_template.png
```

This will:
- Process all PNG images in the directory
- Detect placements in each frame
- Save results to `placement_detections.json`

## Using the Python API

You can also use the detector programmatically:

```python
from detect_placement import PlacementDetector

# Initialize detector
detector = PlacementDetector(
    template_path="clock_template.png",
    threshold=0.7  # Adjust between 0-1 (higher = stricter)
)

# Detect placements
detections = detector.detect_placements(
    "frame_0100.png",
    multi_scale=True  # Try different scales
)

# Results: list of (x, y, confidence)
for x, y, conf in detections:
    print(f"Placement at ({x}, {y}) with confidence {conf:.2f}")

# Visualize
detector.visualize_detections(
    "frame_0100.png",
    detections,
    output_path="result.png"
)
```

## Parameters to Tune

### Threshold (0-1)
- **Higher (0.8-0.95)**: More strict, fewer false positives, may miss some placements
- **Lower (0.6-0.75)**: More lenient, catches more placements, more false positives
- **Default: 0.7** (good starting point)

### Multi-scale Detection
- Enables detection at different scales
- Useful if clock size varies slightly
- Default scales: [0.8, 0.9, 1.0, 1.1, 1.2]

### Template Matching Method
Options (from OpenCV):
- `cv2.TM_CCOEFF_NORMED` (default) - correlation coefficient
- `cv2.TM_CCORR_NORMED` - cross-correlation
- `cv2.TM_SQDIFF_NORMED` - squared difference

## Advanced Usage

### Create Template from Code

```python
detector = PlacementDetector()
detector.create_template_from_roi(
    image_path="frame_0050.png",
    x=320, y=450,  # Top-left of clock
    w=30, h=30,    # Width and height
    save_path="my_clock_template.png"
)
```

### Custom Processing Pipeline

```python
import cv2
from pathlib import Path
from detect_placement import PlacementDetector
import json

detector = PlacementDetector("clock_template.png", threshold=0.75)

# Process video frames
frames_dir = Path("game_replay/frames")
results = []

for frame_path in sorted(frames_dir.glob("frame_*.png")):
    # Extract frame number
    frame_num = int(frame_path.stem.split('_')[1])
    
    # Detect placements
    detections = detector.detect_placements(str(frame_path))
    
    # Store with frame info
    for x, y, conf in detections:
        results.append({
            "frame": frame_num,
            "time_seconds": frame_num / 30.0,  # Assuming 30fps
            "x": x,
            "y": y,
            "confidence": conf
        })

# Save timeline of placements
with open("placement_timeline.json", 'w') as f:
    json.dump(results, f, indent=2)
```

## Troubleshooting

### No detections / Too few detections
- **Lower the threshold**: Try 0.6 or 0.65
- **Enable multi-scale**: Set `multi_scale=True`
- **Check template quality**: Ensure template is clear and not blurry
- **Verify clock is visible**: The clock only appears briefly after placement

### Too many false positives
- **Raise the threshold**: Try 0.8 or 0.85
- **Improve template**: Select a more distinctive clock region
- **Check for similar patterns**: Other UI elements might match

### Detections slightly offset
- Template matching returns top-left corner by default
- The code converts to center points automatically
- If still offset, verify template was extracted correctly

### Performance issues
- **Reduce multi-scale options**: Use fewer scales or just [1.0]
- **Resize images**: Process at lower resolution first
- **Use GPU acceleration**: OpenCV can use CUDA for template matching

## Output Format

The JSON output from batch processing looks like:

```json
{
  "frame_0100.png": [
    {
      "x": 320,
      "y": 450,
      "confidence": 0.87
    },
    {
      "x": 280,
      "y": 520,
      "confidence": 0.82
    }
  ],
  "frame_0150.png": [
    {
      "x": 360,
      "y": 480,
      "confidence": 0.91
    }
  ]
}
```

## Integration with Existing Pipeline

You can combine placement detection with your existing tower detection:

```python
from ultralytics import YOLO
from detect_placement import PlacementDetector

# Load models
tower_model = YOLO("runs/detect/towers_bars_finetune/weights/best.pt")
placement_detector = PlacementDetector("clock_template.png")

# Process frame
frame_path = "frame_0100.png"

# Detect towers and health bars
tower_results = tower_model.predict(frame_path)

# Detect placements
placements = placement_detector.detect_placements(frame_path)

# Combine information
print(f"Frame has {len(tower_results[0].boxes)} towers")
print(f"Frame has {len(placements)} troop placements")
```

## Next Steps

1. **Create training data**: Use detected placements as labels for troop detection
2. **Temporal tracking**: Link placements across frames to track troop movement
3. **Classification**: Add troop type detection (what was placed)
4. **Analysis**: Correlate placements with tower health changes

## References

- OpenCV Template Matching: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
- Multi-scale detection: Essential for handling size variations
- Non-maximum suppression: Removes duplicate detections

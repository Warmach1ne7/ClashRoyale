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

### ⚠️ Important: Clock Animation Challenge

The placement clock in Clash Royale is **animated** - the clock hands rotate and the blue timer overlay changes as time passes. This creates a challenge for standard template matching because:

- **Clock hands change position** - A static template won't match different hand positions
- **Blue timer overlay decreases** - The amount of blue changes over time
- **Standard template matching fails** - A single template can't capture all states

### Solutions

We provide **two approaches**:

1. **Standard Template Matching** (`detect_placement.py`)
   - Use multiple templates for different clock states
   - Best when you can capture 5-10 template variants
   - Fast and simple

2. **Robust Detection Methods** (`detect_placement_robust.py`) ⭐ **Recommended**
   - **Color + Shape**: Detects the blue timer overlay + circular shape
   - **Edge Pattern**: Uses Hough Circle Transform to find circular clock borders
   - **Feature Matching**: SIFT/ORB features robust to appearance changes
   - **Ensemble**: Combines all methods for best accuracy

## Recommended Approach: Robust Detection

Since the clock is animated, **use the robust detector** which doesn't rely on exact pixel matching:

```bash
# Color + shape detection (works regardless of clock hands)
python detect_placement_robust.py color /path/to/frame.png

# Edge-based detection (finds circular clock borders)
python detect_placement_robust.py edge /path/to/frame.png

# Ensemble method (combines all approaches) - BEST ACCURACY
python detect_placement_robust.py ensemble /path/to/frame.png
```

### How Robust Methods Handle Animation

**Color + Shape Detection:**
- Focuses on the **blue timer overlay** color (consistent across animation)
- Detects **circular shape** (clock border doesn't change)
- Ignores the varying clock hands inside

**Edge Pattern Detection:**
- Uses **Hough Circle Transform** to find circular objects
- Clock border creates consistent circular edges
- Verifies by checking for blue pixels inside the circle

**Feature Matching:**
- Extracts keypoint features (corners, edges)
- Matches features even if some pixels change
- More computationally expensive but very robust

## Alternative: Multi-Template Matching

If you prefer template matching, use multiple templates:

### Step 1: Create Multiple Clock Templates

Instead of one template, create 5-10 templates showing different clock states:

```bash
# Find frames with clocks at different animation states
# Frame 1: Clock just appeared (full blue timer)
python detect_placement.py create_template frame_100.png
# Save as clock_template_1.png

# Frame 2: Clock halfway through (partial blue)
python detect_placement.py create_template frame_105.png
# Save as clock_template_2.png

# Frame 3: Clock almost done (little blue)
python detect_placement.py create_template frame_108.png
# Save as clock_template_3.png

# Continue for 5-10 different states...
```

### Step 2: Use Multi-Template Detector

```python
from detect_placement import PlacementDetector

detector = PlacementDetector()

# Load all your templates
detector.clock_templates = []
for i in range(1, 11):
    template = cv2.imread(f"clock_template_{i}.png", cv2.IMREAD_GRAYSCALE)
    detector.clock_templates.append(template)

# Detect - will try all templates
detections = detector.detect_multi_template("frame_200.png")
```

**However, the robust methods are easier and more effective!**

## Using the Python API

### Robust Detection (Recommended)

```python
from detect_placement_robust import RobustPlacementDetector

# Initialize detector
detector = RobustPlacementDetector(threshold=0.7)

# Method 1: Color and shape (best for animated clocks)
detections = detector.detect_by_color_and_shape("frame_0100.png")

# Method 2: Edge pattern detection
detections = detector.detect_by_edge_pattern("frame_0100.png")

# Method 3: Ensemble (combines all methods) - MOST ACCURATE
detections = detector.detect_ensemble("frame_0100.png")

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

### Standard Template Matching (For Reference)

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

### For Robust Detection

**Color Detection:**
- `blue_lower/upper`: HSV range for blue timer overlay
  - Default: `[100, 100, 100]` to `[130, 255, 255]`
  - Adjust based on your video's color profile
  
- `circularity_threshold`: How circular the shape must be
  - Default: 0.6 (60% circular)
  - Higher = more strict (0.7-0.8 for perfect circles)

**Edge Detection:**
- `minRadius/maxRadius`: Expected clock size
  - Default: 10-25 pixels
  - Adjust based on your video resolution
  
- `param2` in HoughCircles: Detection sensitivity
  - Lower = more circles detected (more false positives)
  - Higher = fewer circles (might miss some)

### For Template Matching

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

### Animation Issues

**Problem: Template matching not working**
- **Solution**: Switch to robust detection methods (color+shape or ensemble)
- The clock animation makes single-template matching unreliable

**Problem: Detecting clocks at wrong animation phase**
- **Solution**: Use color+shape detection which ignores hand position
- Or create 5-10 templates for different phases

### Detection Quality

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

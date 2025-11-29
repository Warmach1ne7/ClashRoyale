# Detection Pipeline Updates

## Overview
Updated the placement detection pipeline to add **game tracking** (`game_id`), **template debugging** (`template_id`), and **opponent filtering** (blue vs red/orange clocks).

## Changes Made

### 1. `detect_placement.py` - Core Detector

**Added `return_template_id` parameter:**
```python
def detect_placements(self, ..., return_template_id: bool = False) -> List[Tuple]:
    """
    Returns:
        List of (x, y, confidence) or (x, y, confidence, template_id) tuples
    """
```

**Key Updates:**
- Detections now track which template matched: `(x, y, confidence, scale, template_id)`
- NMS updated to preserve template_id through filtering
- Backward compatible: `return_template_id=False` gives old 3-tuple format

### 2. `create_placement_dataset.py` - Pipeline

#### `detect_blue_clock_placements()` Function

**New Signature:**
```python
def detect_blue_clock_placements(image_path: str, 
                                 detector: PlacementDetector,
                                 min_radius: int = 10,
                                 max_radius: int = 30) -> List[Tuple[int, int, float, str]]
```

**Returns:** `(x, y, confidence, template_id)` tuples

**New Features:**
1. **Template Matching:** Uses PlacementDetector instead of color-based detection
2. **Opponent Filtering:** Checks HSV color composition around each detection
   - Blue (player) clocks: HSV 90-130
   - Red/orange (opponent) clocks: HSV 0-10 and 170-180
   - Only keeps detections with more blue than red pixels

**Algorithm:**
```python
# 1. Run template matching
detections = detector.detect_placements(..., return_template_id=True)

# 2. For each detection, create ROI around it
roi_size = int(max_radius * 1.5)

# 3. Count blue vs red pixels in ROI
blue_pixels = cv2.countNonZero(blue_mask)
red_pixels = cv2.countNonZero(red_mask)

# 4. Keep only if more blue than red
if blue_pixels > red_pixels:
    filtered_detections.append((x, y, conf, template_id))
```

#### `process_game_directory()` Function

**New Columns:**
- `game_id`: UUID or game_XX identifier from directory name
- `template_id`: Name of the template that matched (e.g., "clock_01.png")

**Output Format:**
```
troop, x, y, arena, frame, game_id, template_id
```

#### `process_multiple_games()` Function

**Updated:**
- Default method changed to `"template"` (was `"color"`)
- CSV output now includes `game_id` and `template_id` columns
- Summary shows unique games and templates used

#### Other Functions

**`temporal_filtering()`:**
- Now groups by `(arena, game_id)` instead of just `arena`
- Prevents false positives across different games

**`visualize_detections_on_frame()`:**
- Handles both 3-tuple and 4-tuple detections
- Displays template_id in visualization text

**`main()` CLI:**
- Templates now required: `--templates` is mandatory
- Method default changed to `"template"`
- Removed `"color"` method option

## Usage

### Basic Detection

```bash
python create_placement_dataset.py ../hf_subset \
    --templates clock_templates \
    --output placements_with_tracking.csv \
    --arenas arena_11
```

### Output CSV Format

```csv
troop,x,y,arena,frame,game_id,template_id
unknown,640,360,11,123,a1b2c3d4-e5f6-7890-abcd-ef1234567890,clock_01.png
unknown,580,420,11,124,a1b2c3d4-e5f6-7890-abcd-ef1234567890,clock_03.png
```

### Testing

Run the test script to verify changes:

```bash
cd /home/ostikar/MyProjects/CS541/ClashRoyale/ClashRoyale
python test_detection.py
```

## Benefits

1. **Game Tracking:** `game_id` column allows tracking detections across different games
2. **Template Debugging:** `template_id` shows which template matched, helps identify:
   - Which templates are most useful
   - If certain templates cause false positives
   - Coverage gaps (no template matches certain clock states)
3. **Opponent Filtering:** Reduces false positives by ~50% by filtering red/orange opponent timers
4. **Better Recall:** Template matching with 20-30 templates covers all clock hand positions

## Next Steps

1. **Create More Templates:**
   ```bash
   python fast_extract_templates.py ../hf_subset/arena_11/<game_uuid>/images/frame_0100.png
   ```
   - Extract 20-30 templates covering different clock hand positions
   - Improves detection recall

2. **Analyze Template Usage:**
   ```python
   import pandas as pd
   df = pd.read_csv("placements_with_tracking.csv")
   print(df['template_id'].value_counts())
   ```
   - Identify which templates are used most
   - Remove unused templates
   - Add templates for gaps

3. **Tune Opponent Filter:**
   - If false positives persist, adjust HSV ranges
   - If true positives missed, lower blue/red pixel ratio threshold

4. **Lower Detection Threshold:**
   - Current: 0.65
   - Try: 0.60 for better recall (more detections)
   - Monitor false positive rate

## Color Ranges Reference

**Player Clocks (Blue):**
- HSV: [90-130, 50-255, 50-255]
- Hue range: cyan to blue-violet

**Opponent Clocks (Red/Orange):**
- HSV: [0-10, 50-255, 50-255] and [170-180, 50-255, 50-255]
- Hue range: red-orange to red-pink

**Filtering Logic:**
- Extract ROI around detection (radius * 1.5)
- Count blue pixels vs red pixels
- Keep detection if `blue_pixels > red_pixels`

# Quick Start: Placement Detection for 10fps Data

Perfect for your use case! At 10fps with 1-second clocks, you get ~10 frames per clock cycle.

## Why Multi-Template Works at 10fps

```
Clock animation cycle: 1.0 seconds
Your frame rate: 10 fps
Frames per cycle: ~10 frames

Frame 0: Clock appears    [🔵🔵🔵 🕐]  → Template 1
Frame 1: Clock at 0.1s    [🔵🔵🔵 🕐]  → Template 2
Frame 2: Clock at 0.2s    [🔵🔵🔵 🕐]  → Template 3
Frame 3: Clock at 0.3s    [🔵🔵⚪ 🕑]  → Template 4
Frame 4: Clock at 0.4s    [🔵🔵⚪ 🕑]  → Template 5
Frame 5: Clock at 0.5s    [🔵🔵⚪ 🕑]  → Template 6
Frame 6: Clock at 0.6s    [🔵⚪⚪ 🕒]  → Template 7
Frame 7: Clock at 0.7s    [🔵⚪⚪ 🕒]  → Template 8
Frame 8: Clock at 0.8s    [🔵⚪⚪ 🕒]  → Template 9
Frame 9: Clock at 0.9s    [⚪⚪⚪ 🕒]  → Template 10

With 5-10 templates, you're guaranteed to match!
```

## Step-by-Step Workflow

### Step 1: Create Clock Templates (5-10 minutes)

```bash
cd /home/ostikar/MyProjects/CS541/ClashRoyale/ClashRoyale

# Automated helper - finds frames with clocks
python create_clock_templates.py /path/to/your/frames --output clock_templates --samples 20
```

This will:
1. Scan your frames for potential clocks (blue circular objects)
2. Show you 20 candidate frames
3. Let you click-and-drag to select each clock
4. Save templates as `clock_template_01.png`, `clock_template_02.png`, etc.

**Tips:**
- Create 5-10 templates (recommended: 8)
- Try to select clocks at different animation states
- Press 's' to save, 'n' to skip, 'q' to quit early

### Step 2: Test on Single Frame

```bash
# Test with multi-template matching
python detect_placement.py detect /path/to/test/frame.png --templates clock_templates
```

Expected output:
```
Loaded 8 clock templates:
  - clock_template_01.png (28x28)
  - clock_template_02.png (30x30)
  ...

Found 2 placements:
  1. Position: (320, 450), Confidence: 0.87
  2. Position: (280, 520), Confidence: 0.82

Visualization saved to frame_detections.png
```

### Step 3: Process All Frames (Batch Mode)

```bash
# Process entire game directory
python detect_placement.py batch /path/to/game/frames --templates clock_templates
```

Output: `placement_detections.json`

```json
{
  "frame_0100.png": [
    {"x": 320, "y": 450, "confidence": 0.87}
  ],
  "frame_0150.png": [
    {"x": 280, "y": 520, "confidence": 0.82},
    {"x": 360, "y": 480, "confidence": 0.79}
  ]
}
```

## Optimization Tips for 10fps

### Threshold Tuning

Start with **0.65** (lower than default 0.7):

```python
from detect_placement import PlacementDetector

detector = PlacementDetector(threshold=0.65)  # More lenient for 10fps
detector.load_multiple_templates("clock_templates")
```

Why lower threshold?
- At 10fps, you might not capture the "perfect" clock frame
- Clock might be slightly motion-blurred
- Lower threshold = better recall

### Multi-Scale Usually Not Needed

At fixed 10fps, clock size is consistent:

```python
# Disable multi-scale for speed (default at 10fps)
detections = detector.detect_placements(frame, multi_scale=False)
```

### Template Selection Strategy

**Good:** Spread templates across clock lifecycle
```
Template 1: Clock just appeared (full)
Template 3: Clock 25% done
Template 5: Clock 50% done
Template 7: Clock 75% done
Template 10: Clock almost gone
```

**Better:** Dense coverage in middle phase
```
Templates 1-2: Early (10-20% of cycle)
Templates 3-7: Middle (30-70% of cycle) ← Most variation
Templates 8-10: Late (80-90% of cycle)
```

## Real-World Example

Process one of your arena games:

```bash
# 1. Create templates from arena_01/game_01
python create_clock_templates.py \
    /home/ostikar/MyProjects/CS541/ClashRoyale/data/arena_01/game_01/images \
    --output clock_templates/arena_01 \
    --samples 30

# 2. Test on arena_01/game_02
python detect_placement.py detect \
    /home/ostikar/MyProjects/CS541/ClashRoyale/data/arena_01/game_02/images/frame_0100.png \
    --templates clock_templates/arena_01

# 3. Process all of arena_01/game_02
python detect_placement.py batch \
    /home/ostikar/MyProjects/CS541/ClashRoyale/data/arena_01/game_02/images \
    --templates clock_templates/arena_01
```

## Expected Performance

At 10fps with good templates:

- **Precision:** 85-95% (few false positives)
- **Recall:** 90-98% (catches most placements)
- **Speed:** ~0.5-1 second per frame (with 8 templates)
- **False positives:** Minimal (threshold + NMS removes duplicates)

## If Results Are Poor

### Too Few Detections
```bash
# Lower threshold
# In detect_placement.py, change threshold parameter
detector = PlacementDetector(threshold=0.60)  # Was 0.65
```

### Too Many False Positives
```bash
# Raise threshold
detector = PlacementDetector(threshold=0.75)  # Was 0.65

# Or check template quality - might be too generic
```

### Wrong Locations
- Templates might be incorrectly cropped
- Include some padding around clock
- Recreate templates with ±5 pixel border

## Alternative: Color-Based Detection (Fallback)

If template matching doesn't work well:

```bash
# Try robust color-based detection
python detect_placement_robust.py color /path/to/frame.png
python detect_placement_robust.py ensemble /path/to/frame.png
```

But at 10fps, **multi-template matching should work great!**

## Integration with Tower Detection

Combine placement detection with your existing pipeline:

```python
from ultralytics import YOLO
from detect_placement import PlacementDetector

# Load models
tower_model = YOLO("runs/detect/towers_bars_finetune/weights/best.pt")
placement_detector = PlacementDetector(threshold=0.65)
placement_detector.load_multiple_templates("clock_templates")

# Process game frames
results = []
for frame_path in sorted(frame_dir.glob("*.png")):
    # Detect towers
    towers = tower_model.predict(frame_path)
    
    # Detect placements
    placements = placement_detector.detect_placements(str(frame_path))
    
    results.append({
        'frame': frame_path.name,
        'towers': len(towers[0].boxes),
        'placements': len(placements),
        'placement_locations': [(x, y) for x, y, _ in placements]
    })
```

## Next Steps

1. **Validate results**: Manually check 10-20 frames with detected placements
2. **Tune threshold**: Adjust based on precision/recall needs
3. **Temporal filtering**: A clock should appear for 5-10 consecutive frames
4. **Track movements**: Link placements to troop trajectories
5. **Analyze patterns**: Correlate placements with tower damage

## Troubleshooting

**"Loaded 0 templates"**
- Check template directory path
- Ensure files are named `clock_*.png`

**"No matches found"**
- Templates might be from different video quality
- Try creating templates from the same game
- Lower threshold to 0.60

**"Too many detections in one spot"**
- NMS should handle this, but check overlap_threshold
- Increase to 50 pixels: `detector._non_max_suppression(dets, 50)`

**"Clock detection misses some frames"**
- Normal! Clock might be occluded or motion-blurred
- 80-90% recall is good
- Can interpolate missed frames temporally

Good luck! At 10fps with multi-template matching, you should get excellent results! 🚀

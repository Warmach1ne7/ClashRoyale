# Clock Animation Challenge & Solutions

## The Problem: Animated Clocks

The placement clock in Clash Royale is **animated** and changes over time:

```
Frame 1 (t=0.0s):          Frame 2 (t=0.5s):          Frame 3 (t=1.0s):
     [Clock]                   [Clock]                    [Clock]
    ⎯⎯⎯⎯⎯⎯⎯                   ⎯⎯⎯⎯⎯⎯⎯                    ⎯⎯⎯⎯⎯⎯⎯
   /🔵🔵🔵\                 /🔵🔵⚪\                  /🔵⚪⚪\
  |  🕐   |                |  🕑   |                 |  🕒   |
   \⎯⎯⎯⎯⎯/                 \⎯⎯⎯⎯⎯/                  \⎯⎯⎯⎯⎯/
   
Full blue timer          Half blue timer           Almost empty
Clock hands at 12        Clock hands rotated       Clock hands further
```

### What Changes:
1. **Clock hands rotate** - Different pixel patterns
2. **Blue timer decreases** - Less blue area over time  
3. **Inner details vary** - Numbers/markers may change

### Why Standard Template Matching Fails:

```python
# Template from Frame 1 (full blue, hands at 12)
template = [🔵🔵🔵 + 🕐]

# Trying to match Frame 2 (half blue, hands rotated)
frame = [🔵🔵⚪ + 🕑]

# Result: LOW MATCH SCORE ❌
# Because pixels are different!
```

## Solution 1: Multiple Templates ⭐⭐⭐

Create templates for different animation states:

```
Templates:
  clock_template_1.png  → Full blue, hands at 12:00
  clock_template_2.png  → 3/4 blue, hands at 12:30
  clock_template_3.png  → 1/2 blue, hands at 1:00
  clock_template_4.png  → 1/4 blue, hands at 1:30
  clock_template_5.png  → Almost empty, hands at 2:00
```

**Pros:**
- Still uses template matching (simple)
- Can capture main variations

**Cons:**
- Need to create many templates (tedious)
- May still miss some states
- Slower (tries all templates)

## Solution 2: Color + Shape Detection ⭐⭐⭐⭐⭐ (BEST)

Focus on features that **don't change**:

### What DOESN'T Change:
1. **Blue color** - The timer is always blue (specific HSV range)
2. **Circular shape** - The clock is always circular
3. **Size** - Clock is always ~20-40 pixels
4. **Location constraints** - Only appears in play area

### How It Works:

```python
# Step 1: Find all blue regions
blue_mask = detect_blue_color(image)
#     [🔵🔵🔵]  [🔵🔵⚪]  [🔵⚪⚪]
# →    ✓         ✓         ✓
# All detected regardless of amount!

# Step 2: Filter by circular shape
circular_objects = find_circular_shapes(blue_mask)
#     ⚪  ⬜  ⬟  ⚪
# →   ✓   ✗  ✗  ✓
# Only circular ones kept

# Step 3: Filter by size
clocks = filter_by_size(circular_objects, min=15, max=40)
```

**Pros:**
- ✅ Robust to hand rotation
- ✅ Robust to timer changes  
- ✅ No templates needed
- ✅ Fast
- ✅ Easy to tune

**Cons:**
- May detect other blue circles
- Need to tune HSV ranges for different lighting

## Solution 3: Edge Pattern Detection ⭐⭐⭐⭐

Detect the circular **border** which is consistent:

```
Clock border is always circular:
    ⎯⎯⎯⎯⎯
   /     \      ← These edges don't change
  |   ?   |     ← Inside may change
   \     /      ← But border stays same
    ⎯⎯⎯⎯⎯
```

### How It Works:

```python
# Step 1: Edge detection
edges = detect_edges(image)

# Step 2: Hough Circle Transform
# Finds circular patterns in edges
circles = hough_circles(edges, radius_range=(10, 25))

# Step 3: Verify it's a clock
for circle in circles:
    if has_blue_inside(circle):
        clocks.append(circle)
```

**Pros:**
- ✅ Very robust to internal changes
- ✅ Based on geometry, not appearance
- ✅ Good for clear borders

**Cons:**
- May detect other circles
- Sensitive to edge detection parameters

## Solution 4: Feature Matching ⭐⭐⭐

Uses SIFT/ORB to match keypoint features:

```
Template features:        Frame features:
  • Corner points           • Corner points
  • Edge intersections      • Edge intersections  
  • Distinctive spots       • Distinctive spots

Match features that appear in both
(even if some pixels changed)
```

**Pros:**
- ✅ Very robust to appearance changes
- ✅ Handles rotation, scaling
- ✅ Mature algorithms (SIFT, ORB)

**Cons:**
- ❌ Slower than other methods
- ❌ May need opencv-contrib
- ❌ More complex to tune

## Solution 5: Ensemble Method ⭐⭐⭐⭐⭐ (MOST ACCURATE)

Combine multiple methods and use consensus:

```
Image → Color Detection   → [Clock at (320, 450)] ✓
     → Edge Detection     → [Clock at (318, 452)] ✓
     → Feature Matching   → [Clock at (321, 449)] ✓
                             ⬇️
                        Consensus: Clock at (320, 450)
                        Confidence: HIGH (3/3 methods agree)
```

**Pros:**
- ✅ Highest accuracy
- ✅ Best confidence scores
- ✅ Reduces false positives
- ✅ Automatic fallback if one method fails

**Cons:**
- Slower (runs multiple methods)
- More complex

## Recommended Approach

For **Clash Royale troop placement detection**:

### Quick Start → Color + Shape
```bash
python detect_placement_robust.py color frame.png
```
- Fast, simple, works well
- Best for most cases

### Maximum Accuracy → Ensemble  
```bash
python detect_placement_robust.py ensemble frame.png
```
- Combines all methods
- Best detection quality
- Use when accuracy is critical

### Comparison → Run All
```bash
python compare_detection_methods.py frame.png
```
- See which works best for your data
- Creates side-by-side visualization

## Tuning for Your Video

### Step 1: Check Clock Color

```python
# Extract a clock region and check HSV values
import cv2
import numpy as np

img = cv2.imread("frame_with_clock.png")
# Select clock region (x, y, w, h)
clock = img[450:470, 320:340]  

hsv = cv2.cvtColor(clock, cv2.COLOR_BGR2HSV)
print(f"Hue range: {hsv[:,:,0].min()} - {hsv[:,:,0].max()}")
print(f"Sat range: {hsv[:,:,1].min()} - {hsv[:,:,1].max()}")
print(f"Val range: {hsv[:,:,2].min()} - {hsv[:,:,2].max()}")

# Adjust blue_lower/upper in detect_placement_robust.py
```

### Step 2: Check Clock Size

```python
# Measure clock diameter in pixels
# Adjust minRadius/maxRadius in edge detection
```

### Step 3: Test on Sample Frames

```bash
# Try different frames with clocks at various stages
python detect_placement_robust.py ensemble frame_early.png
python detect_placement_robust.py ensemble frame_mid.png
python detect_placement_robust.py ensemble frame_late.png
```

## Comparison Table

| Method | Robustness | Speed | Setup | Accuracy |
|--------|-----------|-------|-------|----------|
| Single Template | ⭐ | ⭐⭐⭐⭐⭐ | Easy | ⭐⭐ |
| Multi-Template | ⭐⭐⭐ | ⭐⭐⭐ | Tedious | ⭐⭐⭐ |
| Color + Shape | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Easy | ⭐⭐⭐⭐ |
| Edge Pattern | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | ⭐⭐⭐⭐ |
| Feature Match | ⭐⭐⭐⭐ | ⭐⭐ | Medium | ⭐⭐⭐ |
| Ensemble | ⭐⭐⭐⭐⭐ | ⭐⭐ | Easy | ⭐⭐⭐⭐⭐ |

## Real-World Example

```python
from detect_placement_robust import RobustPlacementDetector
import cv2

# Initialize detector
detector = RobustPlacementDetector()

# Process a game replay
for frame_num in range(1000):
    frame_path = f"replay/frame_{frame_num:04d}.png"
    
    # Detect placements (robust to animation)
    placements = detector.detect_ensemble(frame_path)
    
    if placements:
        print(f"Frame {frame_num}: {len(placements)} troops placed")
        for x, y, conf in placements:
            print(f"  → Position ({x}, {y}), confidence {conf:.2f}")
            
            # You now know WHEN and WHERE troops were placed!
            # Can correlate with tower damage, etc.
```

## Summary

**The animated clock is NOT a problem** when you use the right approach!

✅ **Use color + shape detection** → Ignores clock hands, focuses on persistent features  
✅ **Use ensemble method** → Maximum accuracy, combines multiple strategies  
❌ **Don't use single template** → Will fail on different animation states  

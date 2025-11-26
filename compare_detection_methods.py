"""
Compare different detection methods on the same image.
Demonstrates why robust methods work better for animated clocks.
"""

import cv2
import numpy as np
from pathlib import Path
import time
from detect_placement import PlacementDetector
from detect_placement_robust import RobustPlacementDetector


def compare_methods(image_path: str, template_path: str = None):
    """
    Run all detection methods and compare results.
    
    Args:
        image_path: Path to test image
        template_path: Optional template for standard matching
    """
    print("="*70)
    print("CLASH ROYALE PLACEMENT DETECTION - METHOD COMPARISON")
    print("="*70)
    print(f"\nTest image: {image_path}\n")
    
    results = {}
    
    # Method 1: Standard template matching (if template provided)
    if template_path and Path(template_path).exists():
        print("1. Standard Template Matching")
        print("-" * 50)
        try:
            detector1 = PlacementDetector(template_path=template_path, threshold=0.7)
            start = time.time()
            detections1 = detector1.detect_placements(image_path, multi_scale=True)
            elapsed = time.time() - start
            
            results['template'] = {
                'detections': detections1,
                'count': len(detections1),
                'time': elapsed,
                'method': 'Template Matching'
            }
            
            print(f"   Detections: {len(detections1)}")
            print(f"   Time: {elapsed:.3f}s")
            for i, (x, y, conf) in enumerate(detections1[:5]):  # Show first 5
                print(f"   - Position ({x}, {y}), confidence {conf:.3f}")
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            results['template'] = None
    else:
        print("1. Standard Template Matching - SKIPPED (no template)")
        results['template'] = None
    
    # Method 2: Color + Shape
    print("\n2. Color + Shape Detection (Robust to Animation)")
    print("-" * 50)
    try:
        detector2 = RobustPlacementDetector(threshold=0.7)
        start = time.time()
        detections2 = detector2.detect_by_color_and_shape(image_path)
        elapsed = time.time() - start
        
        results['color'] = {
            'detections': detections2,
            'count': len(detections2),
            'time': elapsed,
            'method': 'Color + Shape'
        }
        
        print(f"   Detections: {len(detections2)}")
        print(f"   Time: {elapsed:.3f}s")
        for i, (x, y, conf) in enumerate(detections2[:5]):
            print(f"   - Position ({x}, {y}), confidence {conf:.3f}")
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        results['color'] = None
    
    # Method 3: Edge Pattern
    print("\n3. Edge Pattern Detection (Hough Circles)")
    print("-" * 50)
    try:
        detector3 = RobustPlacementDetector(threshold=0.7)
        start = time.time()
        detections3 = detector3.detect_by_edge_pattern(image_path)
        elapsed = time.time() - start
        
        results['edge'] = {
            'detections': detections3,
            'count': len(detections3),
            'time': elapsed,
            'method': 'Edge Pattern'
        }
        
        print(f"   Detections: {len(detections3)}")
        print(f"   Time: {elapsed:.3f}s")
        for i, (x, y, conf) in enumerate(detections3[:5]):
            print(f"   - Position ({x}, {y}), confidence {conf:.3f}")
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        results['edge'] = None
    
    # Method 4: Ensemble
    print("\n4. Ensemble Method (Combines All)")
    print("-" * 50)
    try:
        detector4 = RobustPlacementDetector(threshold=0.7)
        start = time.time()
        detections4 = detector4.detect_ensemble(image_path)
        elapsed = time.time() - start
        
        results['ensemble'] = {
            'detections': detections4,
            'count': len(detections4),
            'time': elapsed,
            'method': 'Ensemble'
        }
        
        print(f"   Detections: {len(detections4)}")
        print(f"   Time: {elapsed:.3f}s")
        for i, (x, y, conf) in enumerate(detections4[:5]):
            print(f"   - Position ({x}, {y}), confidence {conf:.3f}")
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        results['ensemble'] = None
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for key, result in results.items():
        if result:
            print(f"{result['method']:25s}: {result['count']:2d} detections in {result['time']:.3f}s")
    
    # Create comparison visualization
    print("\nCreating comparison visualization...")
    create_comparison_visual(image_path, results)
    
    return results


def create_comparison_visual(image_path: str, results: dict, output_path: str = None):
    """
    Create a side-by-side visualization comparing all methods.
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    
    h, w = img.shape[:2]
    
    # Create a canvas with 2x2 grid
    canvas = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    
    methods = ['template', 'color', 'edge', 'ensemble']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR
    
    for idx, (method, pos, color) in enumerate(zip(methods, positions, colors)):
        result = results.get(method)
        
        # Copy original image
        img_copy = img.copy()
        
        # Draw detections
        if result and result['detections']:
            for x, y, conf in result['detections']:
                cv2.circle(img_copy, (x, y), 8, color, 2)
                cv2.circle(img_copy, (x, y), 2, (255, 255, 255), -1)
        
        # Add method label
        method_name = result['method'] if result else method.title()
        count = result['count'] if result else 0
        label = f"{method_name}: {count} detections"
        
        cv2.putText(img_copy, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_copy, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        
        # Place in canvas
        y_offset, x_offset = pos
        canvas[y_offset*h:(y_offset+1)*h, x_offset*w:(x_offset+1)*w] = img_copy
    
    # Save
    if output_path is None:
        output_path = image_path.replace('.png', '_comparison.png')
    
    cv2.imwrite(output_path, canvas)
    print(f"Saved comparison to: {output_path}")
    
    return canvas


def batch_compare(frames_dir: str, template_path: str = None, num_frames: int = 10):
    """
    Compare methods across multiple frames and show statistics.
    """
    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob("*.png"))[:num_frames]
    
    print(f"\nBatch comparison across {len(frame_files)} frames...")
    print("="*70)
    
    stats = {
        'template': {'detections': [], 'times': []},
        'color': {'detections': [], 'times': []},
        'edge': {'detections': [], 'times': []},
        'ensemble': {'detections': [], 'times': []}
    }
    
    for i, frame_file in enumerate(frame_files):
        print(f"\nFrame {i+1}/{len(frame_files)}: {frame_file.name}")
        results = compare_methods(str(frame_file), template_path)
        
        # Collect stats
        for method, result in results.items():
            if result:
                stats[method]['detections'].append(result['count'])
                stats[method]['times'].append(result['time'])
    
    # Print aggregate statistics
    print("\n" + "="*70)
    print("AGGREGATE STATISTICS")
    print("="*70)
    
    for method, data in stats.items():
        if data['detections']:
            avg_detections = np.mean(data['detections'])
            avg_time = np.mean(data['times'])
            total_detections = sum(data['detections'])
            
            print(f"\n{method.upper()}:")
            print(f"  Total detections: {total_detections}")
            print(f"  Avg per frame: {avg_detections:.2f}")
            print(f"  Avg time: {avg_time:.3f}s")
            print(f"  Frames with detections: {sum(1 for d in data['detections'] if d > 0)}/{len(data['detections'])}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image:  python compare_detection_methods.py <image_path> [template_path]")
        print("  Batch:         python compare_detection_methods.py batch <frames_dir> [template_path] [num_frames]")
        sys.exit(1)
    
    if sys.argv[1] == "batch":
        if len(sys.argv) < 3:
            print("Error: Provide frames directory")
            sys.exit(1)
        
        frames_dir = sys.argv[2]
        template_path = sys.argv[3] if len(sys.argv) > 3 else None
        num_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        
        batch_compare(frames_dir, template_path, num_frames)
    
    else:
        image_path = sys.argv[1]
        template_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        compare_methods(image_path, template_path)

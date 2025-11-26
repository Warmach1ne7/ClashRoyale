"""
Robust troop placement detection that handles the animated clock icon.
Uses multiple strategies to detect clocks despite changing hands and timer overlay.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json


class RobustPlacementDetector:
    """
    Detect troop placements using methods robust to clock animation.
    Handles changing clock hands and blue timer overlay.
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize the robust placement detector.
        
        Args:
            threshold: Detection threshold (0-1)
        """
        self.threshold = threshold
        self.clock_templates = []  # Multiple templates for different clock states
        self.clock_size_range = (20, 50)  # Expected clock size range in pixels
    
    def load_multiple_templates(self, template_dir: str):
        """
        Load multiple clock templates showing different animation states.
        
        Args:
            template_dir: Directory containing clock template images
        """
        template_path = Path(template_dir)
        self.clock_templates = []
        
        for img_path in sorted(template_path.glob("clock_*.png")):
            template = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.clock_templates.append(template)
        
        print(f"Loaded {len(self.clock_templates)} clock templates")
    
    def detect_by_color_and_shape(self, image_path: str) -> List[Tuple[int, int, float]]:
        """
        Detect clocks using color segmentation + circular shape detection.
        This is robust to clock hand animation since it focuses on the clock border/face.
        
        Args:
            image_path: Path to the game frame
        
        Returns:
            List of (x, y, confidence) tuples
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Clock characteristics in Clash Royale:
        # - Usually has a white/light circular border
        # - Blue timer overlay (specific blue color)
        # - Dark center with clock hands
        
        # Detect blue timer overlay (this is consistent)
        # Blue hue range in HSV
        blue_lower = np.array([100, 100, 100])  # Adjust based on actual clock color
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Also detect white/light colors (clock border)
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(blue_mask, white_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find circular contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (clocks are relatively small)
            if area < 100 or area > 2000:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Clocks should be relatively circular (0.7+ is good)
            if circularity > 0.6:
                # Get center point
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Use circularity as confidence
                    confidence = circularity
                    detections.append((cx, cy, confidence))
        
        return detections
    
    def detect_by_edge_pattern(self, image_path: str) -> List[Tuple[int, int, float]]:
        """
        Detect clocks by looking for circular edge patterns.
        Clock border creates a consistent circular edge regardless of hand position.
        
        Args:
            image_path: Path to the game frame
        
        Returns:
            List of (x, y, confidence) tuples
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,  # Minimum distance between circles
            param1=50,
            param2=15,   # Lower = more circles detected
            minRadius=10,  # Minimum clock radius
            maxRadius=25   # Maximum clock radius
        )
        
        detections = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for circle in circles[0, :]:
                cx, cy, radius = circle
                
                # Verify this is likely a clock by checking color in the region
                roi = img[max(0, cy-radius):min(img.shape[0], cy+radius),
                         max(0, cx-radius):min(img.shape[1], cx+radius)]
                
                if roi.size == 0:
                    continue
                
                # Check if region has blue/white colors typical of clocks
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                blue_mask = cv2.inRange(hsv_roi, np.array([100, 50, 50]), np.array([130, 255, 255]))
                blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
                
                if blue_ratio > 0.1:  # At least 10% blue pixels
                    confidence = min(blue_ratio * 2, 1.0)  # Scale to 0-1
                    detections.append((int(cx), int(cy), confidence))
        
        return detections
    
    def detect_by_feature_matching(self, image_path: str, 
                                   template_path: str = None) -> List[Tuple[int, int, float]]:
        """
        Use feature matching (SIFT/ORB) which is more robust to appearance changes.
        This can match key features even if clock hands move.
        
        Args:
            image_path: Path to the game frame
            template_path: Path to a reference clock template
        
        Returns:
            List of (x, y, confidence) tuples
        """
        if template_path is None or not Path(template_path).exists():
            return []
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or template is None:
            return []
        
        # Initialize ORB detector (SIFT requires opencv-contrib)
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(img, None)
        
        if des1 is None or des2 is None:
            return []
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Group matches by location to find clock instances
        detections = []
        
        if len(matches) > 10:  # Need enough matches
            # Get matched keypoint locations in the image
            matched_points = [kp2[m.trainIdx].pt for m in matches[:50]]
            
            # Cluster nearby points
            from sklearn.cluster import DBSCAN
            import numpy as np
            
            if len(matched_points) > 3:
                clustering = DBSCAN(eps=30, min_samples=3).fit(matched_points)
                
                for label in set(clustering.labels_):
                    if label == -1:  # Noise
                        continue
                    
                    cluster_points = [matched_points[i] for i in range(len(matched_points)) 
                                    if clustering.labels_[i] == label]
                    
                    # Calculate center of cluster
                    cx = int(np.mean([p[0] for p in cluster_points]))
                    cy = int(np.mean([p[1] for p in cluster_points]))
                    
                    # Confidence based on cluster size
                    confidence = min(len(cluster_points) / 20.0, 1.0)
                    detections.append((cx, cy, confidence))
        
        return detections
    
    def detect_multi_template(self, image_path: str) -> List[Tuple[int, int, float]]:
        """
        Use multiple templates representing different clock states.
        Returns the best matches across all templates.
        
        Args:
            image_path: Path to the game frame
        
        Returns:
            List of (x, y, confidence) tuples
        """
        if not self.clock_templates:
            raise ValueError("No templates loaded. Call load_multiple_templates() first.")
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        all_detections = []
        
        # Try each template
        for template in self.clock_templates:
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            
            # Find locations above threshold
            locations = np.where(result >= self.threshold)
            
            h, w = template.shape
            for pt in zip(*locations[::-1]):
                confidence = result[pt[1], pt[0]]
                cx = pt[0] + w // 2
                cy = pt[1] + h // 2
                all_detections.append((cx, cy, confidence))
        
        # Apply NMS
        return self._non_max_suppression(all_detections)
    
    def detect_ensemble(self, image_path: str) -> List[Tuple[int, int, float]]:
        """
        Combine multiple detection methods for robust results.
        Uses voting/consensus across methods.
        
        Args:
            image_path: Path to the game frame
        
        Returns:
            List of (x, y, confidence) tuples
        """
        all_detections = []
        
        # Method 1: Color and shape
        try:
            detections1 = self.detect_by_color_and_shape(image_path)
            all_detections.extend([(x, y, conf, "color") for x, y, conf in detections1])
        except Exception as e:
            print(f"Color detection failed: {e}")
        
        # Method 2: Edge pattern
        try:
            detections2 = self.detect_by_edge_pattern(image_path)
            all_detections.extend([(x, y, conf, "edge") for x, y, conf in detections2])
        except Exception as e:
            print(f"Edge detection failed: {e}")
        
        # Method 3: Multi-template (if templates available)
        if self.clock_templates:
            try:
                detections3 = self.detect_multi_template(image_path)
                all_detections.extend([(x, y, conf, "template") for x, y, conf in detections3])
            except Exception as e:
                print(f"Template detection failed: {e}")
        
        # Cluster detections from different methods
        final_detections = self._cluster_consensus(all_detections)
        
        return final_detections
    
    def _cluster_consensus(self, detections: List[Tuple[int, int, float, str]], 
                          distance_threshold: int = 30) -> List[Tuple[int, int, float]]:
        """
        Cluster detections from multiple methods and use consensus.
        Detections supported by multiple methods get higher confidence.
        """
        if not detections:
            return []
        
        # Simple clustering by distance
        clusters = []
        
        for det in detections:
            x, y, conf, method = det
            
            # Find if this belongs to existing cluster
            found_cluster = False
            for cluster in clusters:
                # Check distance to cluster center
                cx, cy = cluster['center']
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                
                if dist < distance_threshold:
                    cluster['points'].append((x, y, conf, method))
                    found_cluster = True
                    break
            
            if not found_cluster:
                clusters.append({
                    'center': (x, y),
                    'points': [(x, y, conf, method)]
                })
        
        # Convert clusters to final detections
        final_detections = []
        
        for cluster in clusters:
            points = cluster['points']
            
            # Calculate weighted center
            total_conf = sum(p[2] for p in points)
            if total_conf == 0:
                continue
            
            cx = sum(p[0] * p[2] for p in points) / total_conf
            cy = sum(p[1] * p[2] for p in points) / total_conf
            
            # Boost confidence if multiple methods agree
            method_count = len(set(p[3] for p in points))
            avg_conf = total_conf / len(points)
            boosted_conf = min(avg_conf * (1 + 0.2 * (method_count - 1)), 1.0)
            
            final_detections.append((int(cx), int(cy), boosted_conf))
        
        return sorted(final_detections, key=lambda x: x[2], reverse=True)
    
    def _non_max_suppression(self, detections: List[Tuple[int, int, float]], 
                            overlap_threshold: int = 30) -> List[Tuple[int, int, float]]:
        """Remove overlapping detections."""
        if not detections:
            return []
        
        detections = sorted(detections, key=lambda x: x[2], reverse=True)
        kept = []
        
        for det in detections:
            x, y, conf = det[:3]  # Handle both 3 and 4-tuple inputs
            
            is_duplicate = False
            for kx, ky, kconf in kept:
                distance = np.sqrt((x - kx)**2 + (y - ky)**2)
                if distance < overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append((x, y, conf))
        
        return kept
    
    def visualize_detections(self, image_path: str, detections: List[Tuple[int, int, float]], 
                            output_path: str = None, show: bool = False):
        """Visualize detected placements."""
        img = cv2.imread(image_path)
        
        for x, y, conf in detections:
            cv2.circle(img, (x, y), 10, (0, 255, 0), 2)
            cv2.line(img, (x-15, y), (x+15, y), (0, 255, 0), 2)
            cv2.line(img, (x, y-15), (x, y+15), (0, 255, 0), 2)
            cv2.putText(img, f"{conf:.2f}", (x+15, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Saved visualization to {output_path}")
        
        if show:
            cv2.imshow("Placement Detections", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return img


def create_multiple_clock_templates(video_frames_dir: str, output_dir: str):
    """
    Helper to extract multiple clock templates from different frames.
    Creates templates showing different clock animation states.
    
    Usage:
        1. Identify frames with visible clocks
        2. This will help you extract multiple template variants
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    frames_path = Path(video_frames_dir)
    frame_files = list(frames_path.glob("*.png"))[:50]  # Check first 50 frames
    
    print(f"Scanning {len(frame_files)} frames for clocks...")
    print("Click on each clock you see, press 's' to save, 'n' for next frame, 'q' to quit")
    
    template_count = 0
    
    for frame_file in frame_files:
        img = cv2.imread(str(frame_file))
        clone = img.copy()
        
        roi_selected = False
        start_point = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal start_point, roi_selected, img
            
            if event == cv2.EVENT_LBUTTONDOWN:
                start_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and start_point:
                img = clone.copy()
                cv2.rectangle(img, start_point, (x, y), (0, 255, 0), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                end_point = (x, y)
                roi_selected = True
                cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
        
        cv2.namedWindow(frame_file.name)
        cv2.setMouseCallback(frame_file.name, mouse_callback)
        
        while True:
            cv2.imshow(frame_file.name, img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and roi_selected and start_point:
                # Save template
                x1 = min(start_point[0], start_point[0])
                y1 = min(start_point[1], start_point[1])
                # Get current mouse position as end point
                # For simplicity, use a fixed size or last drawn rectangle
                # This is a simplified version
                pass
            elif key == ord('n'):
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Color-based:  python detect_placement_robust.py color <image_path>")
        print("  Edge-based:   python detect_placement_robust.py edge <image_path>")
        print("  Ensemble:     python detect_placement_robust.py ensemble <image_path>")
        sys.exit(1)
    
    method = sys.argv[1]
    image_path = sys.argv[2]
    
    detector = RobustPlacementDetector(threshold=0.7)
    
    if method == "color":
        detections = detector.detect_by_color_and_shape(image_path)
    elif method == "edge":
        detections = detector.detect_by_edge_pattern(image_path)
    elif method == "ensemble":
        detections = detector.detect_ensemble(image_path)
    else:
        print(f"Unknown method: {method}")
        sys.exit(1)
    
    print(f"\nFound {len(detections)} placements using {method} method:")
    for i, (x, y, conf) in enumerate(detections):
        print(f"  {i+1}. Position: ({x}, {y}), Confidence: {conf:.3f}")
    
    # Save visualization
    output = image_path.replace('.png', f'_detections_{method}.png')
    detector.visualize_detections(image_path, detections, output_path=output)
    print(f"\nVisualization saved to {output}")

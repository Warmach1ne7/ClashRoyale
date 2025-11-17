"""
Remove false health_bar labels from YOLO txts when the bar isn't visible.
Heuristic: HSV color segmentation (green/yellow/red) + horizontal fill check.
"""
import cv2
import numpy as np
from pathlib import Path

HEALTH_BAR_CLASS_ID = 4  # adjust if needed

def yolo_to_xyxy(line, w, h):
    parts = line.strip().split()
    if len(parts) < 5: return None
    cls = int(parts[0]); xc, yc, bw, bh = map(float, parts[1:5])
    x1 = int((xc - bw/2) * w); y1 = int((yc - bh/2) * h)
    x2 = int((xc + bw/2) * w); y2 = int((yc + bh/2) * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    return cls, x1, y1, x2, y2

def bar_present(bar_roi: np.ndarray, min_col_fill=0.30, min_run_frac=0.20) -> bool:
    """
    min_col_fill: a column counts as 'filled' if >=30% pixels are bar-colored
    min_run_frac: require a continuous filled run >=20% of ROI width
    """
    if bar_roi.size == 0: return False
    hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)

    # Green
    g = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    # Yellow
    y = cv2.inRange(hsv, np.array([20, 70, 70]), np.array([35, 255, 255]))
    # Red (two ranges)
    r1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([10, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([170, 70, 70]), np.array([180, 255, 255]))
    mask = g | y | r1 | r2

    h, w = mask.shape
    col_sums = np.sum(mask > 0, axis=0)
    filled = (col_sums >= h * min_col_fill)

    # longest continuous run of True
    run, best = 0, 0
    for v in filled:
        run = run + 1 if v else 0
        best = max(best, run)

    return best >= int(w * min_run_frac)

def process_image(img_path: Path, lbl_path: Path) -> int:
    """
    Returns number of bar labels removed for this image.
    """
    if not lbl_path.exists():
        return 0
    image = cv2.imread(str(img_path))
    if image is None:
        return 0

    h, w = image.shape[:2]
    lines = [ln for ln in lbl_path.read_text().strip().splitlines() if ln.strip()]
    keep = []
    removed = 0

    for ln in lines:
        parsed = yolo_to_xyxy(ln, w, h)
        if not parsed:
            continue
        cls, x1, y1, x2, y2 = parsed
        if cls != HEALTH_BAR_CLASS_ID:
            keep.append(ln)
            continue

        roi = image[y1:y2, x1:x2]
        if bar_present(roi):
            keep.append(ln)
        else:
            removed += 1

    if removed > 0:
        if keep:
            Path(lbl_path).write_text('\n'.join(keep) + '\n')
        else:
            # empty label file: write nothing but keep file for YOLO compatibility
            Path(lbl_path).write_text('')
    return removed

def main():
    root = Path('/home/ostikar/MyProjects/CS541/ClashRoyale/data')
    arenas = [root / f'arena_{i:02d}' for i in range(1, 11)]
    total_imgs, total_removed = 0, 0

    for arena in arenas:
        if not arena.exists(): continue
        for game in sorted([d for d in arena.iterdir() if d.is_dir() and d.name.startswith('game_')]):
            images = (game / 'images')
            labels = (game / 'labels')
            if not images.exists() or not labels.exists(): continue

            for img_path in sorted(list(images.glob('*.png')) + list(images.glob('*.jpg'))):
                lbl_path = labels / f'{img_path.stem}.txt'
                total_imgs += 1
                total_removed += process_image(img_path, lbl_path)

    print(f'Checked images: {total_imgs}')
    print(f'Removed false health_bar labels: {total_removed}')
    print('Done. Re-run your split and train.')

if __name__ == '__main__':
    main()
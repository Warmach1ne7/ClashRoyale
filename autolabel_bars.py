"""
Append health_bar labels from bar_rois.json to existing YOLO labels.
Skips king bars when JSON has null (not visible at full health).
"""
import json
from pathlib import Path
from typing import List, Tuple

HEALTH_BAR_CLASS_ID = 4  # Update if your data.yaml maps differently

def roi_to_yolo(roi: List[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = roi
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return xc, yc, w, h

def main():
    data_root = Path('/home/ostikar/MyProjects/CS541/ClashRoyale/data')
    bar_json = data_root / 'towers3cls' / 'bar_rois.json'

    with open(bar_json, 'r') as f:
        bar_rois = json.load(f)

    # Map json keys to consistent tower ids (so we know which label file to edit)
    # Here we assume bars are at static normalized positions for all frames.
    # We'll simply add one health_bar entry to every image's label for the towers that have a bar ROI.
    keys = [
        'princess_top_l_bar', 'princess_top_r_bar',
        'princess_bot_l_bar', 'princess_bot_r_bar'
    ]

    arenas = [data_root / f'arena_{i:02d}' for i in range(1, 11)]
    added = 0
    total_images = 0

    for arena in arenas:
        if not arena.exists():
            continue
        for game_dir in sorted([d for d in arena.iterdir() if d.is_dir() and d.name.startswith('game_')]):
            images_dir = game_dir / 'images'
            labels_dir = game_dir / 'labels'
            if not images_dir.exists():
                continue
            labels_dir.mkdir(exist_ok=True)

            image_files = sorted(images_dir.glob('*.png')) + sorted(images_dir.glob('*.jpg'))
            for img_path in image_files:
                total_images += 1
                label_path = labels_dir / f'{img_path.stem}.txt'
                lines = []
                if label_path.exists():
                    lines = label_path.read_text().strip().splitlines()

                # Append bar boxes for the towers that have them defined
                for k in keys:
                    roi = bar_rois.get(k, None)
                    if not roi:
                        continue  # skip null or missing (e.g., king bars)
                    xc, yc, w, h = roi_to_yolo(roi)
                    # Basic sanity filter to avoid duplicates: don't double-add if already present
                    yolo_line = f"{HEALTH_BAR_CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                    if yolo_line not in lines:
                        lines.append(yolo_line)
                        added += 1

                if lines:
                    label_path.write_text('\n'.join(lines) + '\n')

    print(f"Processed images: {total_images}")
    print(f"Health bar labels appended: {added}")
    print("Done. Re-run your split to refresh the consolidated dataset.")

if __name__ == '__main__':
    main()
import json
from pathlib import Path
from typing import Dict, List, Tuple

CLASS_MAP = {'king': 0, 'princess': 1}
ROI_ORDER = [
    "king_top",
    "king_bottom",
    "princess_top_l",
    "princess_top_r",
    "princess_bot_l",
    "princess_bot_r"
]

def roi_to_yolo(roi: List[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = roi
    return ( (x1 + x2) / 2,
             (y1 + y2) / 2,
             abs(x2 - x1),
             abs(y2 - y1) )

def load_destruction(game_dir: Path) -> Dict[str, int]:
    path = game_dir / "destruction.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    # Normalize: None or missing -> not destroyed
    destr = {}
    for k in ROI_ORDER:
        v = raw.get(k, None)
        if v is None:
            continue
        destr[k] = int(v)
    return destr

def process_game(game_dir: Path, rois: Dict[str, List[float]]):
    images_dir = game_dir / "images"
    if not images_dir.exists():
        return 0, 0
    frames = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
    if not frames:
        return 0, 0
    destruction = load_destruction(game_dir)
    labels_dir = game_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    written = 0
    for idx, img_path in enumerate(frames):
        lines = []
        for roi_name in ROI_ORDER:
            if roi_name not in rois:
                continue
            roi = rois[roi_name]
            # zero-area skip
            if roi[0] == roi[2] or roi[1] == roi[3]:
                continue
            # If destruction frame recorded and we are at/after it, skip
            destroy_frame = destruction.get(roi_name, None)
            if destroy_frame is not None and idx >= destroy_frame:
                continue
            cls_id = CLASS_MAP['king'] if roi_name.startswith('king') else CLASS_MAP['princess']
            x_c, y_c, w, h = roi_to_yolo(roi)
            lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        if lines:
            out = labels_dir / f"{img_path.stem}.txt"
            with open(out, "w") as f:
                f.write("\n".join(lines) + "\n")
            written += 1
    return len(frames), written

def main():
    data_root = Path("/home/ostikar/MyProjects/CS541/ClashRoyale/data")
    rois_json = data_root / "towers" / "rois.json"
    with open(rois_json, "r") as f:
        rois = json.load(f)

    arenas = []
    for i in range(1, 11):
        a = data_root / f"arena_{i:02d}"
        if a.exists():
            arenas.append(a)
    total_frames = 0
    total_labelled = 0
    for arena in arenas:
        print(f"\nArena {arena.name}")
        games = sorted([d for d in arena.iterdir() if d.is_dir() and d.name.startswith("game_")])
        for game in games:
            frames, labelled = process_game(game, rois)
            total_frames += frames
            total_labelled += labelled
            print(f"  {game.name}: frames={frames} labelled={labelled}")
    print("\n====================")
    print(f"Total frames: {total_frames}")
    print(f"Frames with labels written: {total_labelled}")
    print("====================")
    print("Note: Destroyed tower frames excluded based on destruction.json per game.")

if __name__ == "__main__":
    main()
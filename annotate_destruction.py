import json
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

ROI_NAMES = [
    "king_top",
    "king_bottom",
    "princess_top_l",
    "princess_top_r",
    "princess_bot_l",
    "princess_bot_r"
]

def list_frames(images_dir: Path) -> List[Path]:
    frames = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
    return frames

def prompt_frame(name: str, total: int) -> Optional[int]:
    while True:
        val = input(f"Destroyed frame for {name} (0-{total-1}, blank if never): ").strip()
        if val == "":
            return None
        if val.isdigit():
            num = int(val)
            if 0 <= num < total:
                return num
        print("Invalid input; try again.")

def main():
    data_root = Path("/home/ostikar/MyProjects/CS541/ClashRoyale/data")
    arena = input("Arena (e.g. 01): ").strip()
    game = input("Game (e.g. 1): ").strip()
    game_dir = data_root / f"arena_{int(arena):02d}" / f"game_{game}"
    images_dir = game_dir / "images"
    assert images_dir.exists(), f"No images directory at {images_dir}"
    frames = list_frames(images_dir)
    print(f"Found {len(frames)} frames.")

    destruction = {}
    for roi in ROI_NAMES:
        destruction[roi] = prompt_frame(roi, len(frames))

    out_path = game_dir / "destruction.json"
    with open(out_path, "w") as f:
        json.dump(destruction, f, indent=2)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
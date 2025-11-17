import json
from pathlib import Path
import cv2
import shutil

RAW_DIR = Path(r"/home/ostikar/MyProjects/CS541/ClashRoyale/data/arena_02/game_01")
OUT_DIR = Path(r"/home/ostikar/MyProjects/CS541/ClashRoyale/data/towers")
IMG_OUT = OUT_DIR / "images"
LBL_OUT = OUT_DIR / "labels"
OUT_DIR.mkdir(parents=True, exist_ok=True); IMG_OUT.mkdir(exist_ok=True); LBL_OUT.mkdir(exist_ok=True)

DEFAULT_ROIS = {
    "king_top": [0.5777777777777777, 0.2989583333333333, 0.5777777777777777, 0.2989583333333333],
    "princess_top_l": [0.18333333333333332, 0.2604166666666667, 0.3296296296296296, 0.35833333333333334],
    "princess_top_r": [0.6722222222222223, 0.25625, 0.8185185185185185, 0.36041666666666666],
    "king_bottom": [0.43333333333333335, 0.7270833333333333, 0.5722222222222222, 0.8677083333333333],
    "princess_bot_l": [0.18703703703703703, 0.6645833333333333, 0.5740740740740741, 0.8666666666666667],
    "princess_bot_r": [0.3333333333333333, 0.6645833333333333, 0.674074074074074, 0.7458333333333333],
}
CLASS_ID = {"king": 0, "princess": 1}
ROIS_PATH = OUT_DIR / "rois.json"

def load_images(folder: Path):
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            yield p

def rect_to_yolo(x1, y1, x2, y2, W, H):
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w  = abs(x2 - x1) / W
    h  = abs(y2 - y1) / H
    return cx, cy, w, h

def norm_rect_to_abs(r, W, H):
    x1 = int(r[0] * W); y1 = int(r[1] * H); x2 = int(r[2] * W); y2 = int(r[3] * H)
    return x1, y1, x2, y2

def interactive_draw(reference_img_path: Path):
    img = cv2.imread(str(reference_img_path))
    H, W = img.shape[:2]
    rois = {}
    order = [
        ("king_top", (255, 200, 0), "Draw TOP KING"),
        ("princess_top_l", (0, 200, 255), "Draw TOP LEFT PRINCESS"),
        ("princess_top_r", (0, 200, 255), "Draw TOP RIGHT PRINCESS"),
        ("king_bottom", (255, 200, 0), "Draw BOTTOM KING"),
        ("princess_bot_l", (0, 200, 255), "Draw BOTTOM LEFT PRINCESS"),
        ("princess_bot_r", (0, 200, 255), "Draw BOTTOM RIGHT PRINCESS"),
    ]

    drawing = {"start": None, "end": None, "current": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["start"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing["start"] is not None:
            drawing["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing["end"] = (x, y)

    cv2.namedWindow("Draw ROIs"); cv2.setMouseCallback("Draw ROIs", on_mouse)

    for name, color, prompt in order:
        cur = img.copy()
        while True:
            disp = cur.copy()
            if drawing["start"] and drawing["current"]:
                x1,y1 = drawing["start"]; x2,y2 = drawing["current"]
                cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
            cv2.putText(disp, f"{prompt}: drag box, ENTER to confirm, R to reset, ESC to abort",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Draw ROIs", disp)
            k = cv2.waitKey(20) & 0xFF
            if k == 13 and drawing["start"] and (drawing["end"] or drawing["current"]):  # ENTER
                x1,y1 = drawing["start"]; x2,y2 = drawing["end"] or drawing["current"]
                x1,x2 = sorted([x1,x2]); y1,y2 = sorted([y1,y2])
                rois[name] = [x1/W, y1/H, x2/W, y2/H]
                drawing["start"] = drawing["end"] = drawing["current"] = None
                break
            elif k in (ord('r'), ord('R')):
                drawing["start"] = drawing["end"] = drawing["current"] = None
            elif k == 27:  # ESC
                cv2.destroyAllWindows()
                raise SystemExit("Aborted.")
        # visualize accumulated boxes
        cur = img.copy()
        for n, r in rois.items():
            a,b,c,d = norm_rect_to_abs(r, W, H)
            cv2.rectangle(cur, (a,b), (c,d), (0,255,0), 2)
            cv2.putText(cur, n, (a, max(0,b-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.destroyAllWindows()
    return rois

def ensure_rois():
    if ROIS_PATH.exists():
        return json.loads(ROIS_PATH.read_text())
    # Pick first image as reference
    ref = next(load_images(RAW_DIR), None)
    if ref is None:
        raise FileNotFoundError("No images found in RAW_DIR")
    try:
        rois = interactive_draw(ref)
    except Exception:
        # fallback to defaults if UI not used
        rois = DEFAULT_ROIS
    ROIS_PATH.write_text(json.dumps(rois, indent=2))
    return rois

def main():
    rois = ensure_rois()
    # Map ROI names to classes
    name_to_cls = {
        "king_top": CLASS_ID["king"],
        "king_bottom": CLASS_ID["king"],
        "princess_top_l": CLASS_ID["princess"],
        "princess_top_r": CLASS_ID["princess"],
        "princess_bot_l": CLASS_ID["princess"],
        "princess_bot_r": CLASS_ID["princess"],
    }
    for img_path in load_images(RAW_DIR):
        img = cv2.imread(str(img_path)); H, W = img.shape[:2]
        # Copy image to dataset folder
        out_img = IMG_OUT / img_path.name
        shutil.copy2(img_path, out_img)
        # Write YOLO label
        lines = []
        for name, rect in rois.items():
            x1,y1,x2,y2 = norm_rect_to_abs(rect, W, H)
            cx, cy, w, h = rect_to_yolo(x1,y1,x2,y2, W, H)
            cls_id = name_to_cls[name]
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (LBL_OUT / (img_path.stem + ".txt")).write_text("\n".join(lines))

if __name__ == "__main__":
    main()
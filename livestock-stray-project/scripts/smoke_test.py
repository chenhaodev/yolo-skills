"""
牛马占道 Smoke Test — Circle annotation visualizer
Draws circles (not boxes) around detected livestock with confidence scores.
Usage: python scripts/smoke_test.py
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/pretrained/yolov8m.pt"
TEST_DIR    = Path("data/smoke-test")
OUT_DIR     = Path("data/smoke-test/results_circle")
CONF_THRESH = 0.25

# COCO class IDs we care about
TARGET = {17: "horse", 19: "cattle"}

# Visual style
CIRCLE_COLOR  = (0, 255, 80)    # bright green for livestock
NEG_COLOR     = (180, 180, 180) # grey overlay for negatives
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.9
FONT_THICK    = 2

# ── Helpers ─────────────────────────────────────────────────────────────────

def draw_circle_annotation(img: np.ndarray, x1: float, y1: float,
                            x2: float, y2: float,
                            label: str, conf: float) -> np.ndarray:
    """Draw an ellipse fitted to the bounding box + label with confidence."""
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    rx = int((x2 - x1) / 2)
    ry = int((y2 - y1) / 2)

    # Outer ellipse (thick, semi-transparent fill)
    overlay = img.copy()
    cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360, CIRCLE_COLOR, -1)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    # Ellipse border
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, CIRCLE_COLOR, 3)

    # Label badge
    text = f"{label} {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    badge_x = cx - tw // 2
    badge_y = max(cy - ry - 10, th + 4)
    cv2.rectangle(img,
                  (badge_x - 4, badge_y - th - 4),
                  (badge_x + tw + 4, badge_y + baseline),
                  CIRCLE_COLOR, -1)
    cv2.putText(img, text,
                (badge_x, badge_y),
                FONT, FONT_SCALE, (0, 0, 0), FONT_THICK, cv2.LINE_AA)
    return img


def stamp_status(img: np.ndarray, alert: bool, count: int) -> np.ndarray:
    """Stamp ALERT or CLEAR banner in top-right corner."""
    h, w = img.shape[:2]
    if alert:
        text  = f"ALERT  {count} animal(s)"
        color = (0, 60, 255)
    else:
        text  = "CLEAR"
        color = (60, 200, 60)

    (tw, th), _ = cv2.getTextSize(text, FONT, 1.2, 3)
    x = w - tw - 20
    y = th + 20
    cv2.rectangle(img, (x - 10, y - th - 10), (x + tw + 10, y + 10), color, -1)
    cv2.putText(img, text, (x, y), FONT, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    return img


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)

    images = sorted(TEST_DIR.glob("*.jpg"))
    # Interleave: sort so negatives (neg_*) and positives alternate
    positives = [p for p in images if not p.name.startswith("neg_")]
    negatives = [p for p in images if p.name.startswith("neg_")]
    mixed = []
    for i in range(max(len(positives), len(negatives))):
        if i < len(positives):
            mixed.append(positives[i])
        if i < len(negatives):
            mixed.append(negatives[i])

    print("=" * 65)
    print("  牛马占道 Smoke Test — Mixed Samples, Circle Annotation")
    print("=" * 65)
    print(f"  {'Image':<35} {'Result':<10} {'Detections'}")
    print("  " + "-" * 62)

    for img_path in mixed:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  SKIP {img_path.name} (unreadable)")
            continue

        # Resize large images for display (keep aspect ratio)
        max_dim = 1280
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        results = model(img, conf=CONF_THRESH, verbose=False)
        r = results[0]

        livestock_found = []
        for box in r.boxes:
            cls_id = int(box.cls)
            if cls_id not in TARGET:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf)
            label = TARGET[cls_id]
            livestock_found.append((label, conf, x1, y1, x2, y2))
            draw_circle_annotation(img, x1, y1, x2, y2, label, conf)

        alert = len(livestock_found) > 0
        stamp_status(img, alert, len(livestock_found))

        out_path = OUT_DIR / img_path.name
        cv2.imwrite(str(out_path), img)

        status = "ALERT 🚨" if alert else "clear  ✓"
        det_str = ", ".join(f"{l}({c:.2f})" for l, c, *_ in livestock_found) or "—"
        print(f"  {img_path.name:<35} {status:<10} {det_str}")

    print("=" * 65)
    print(f"\n  Annotated images saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()

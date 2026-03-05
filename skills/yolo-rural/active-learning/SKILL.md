---
name: yolo-rural:active-learning
description: >
  Active learning loop for rural CV: collect low-confidence edge samples, annotate
  with SAM assistance, merge into dataset, retrain, compare, redeploy. Use when user
  wants to improve model with new data, retrain on recent edge failures, or run the
  annotation → train → deploy cycle. Triggered by: active learning, new data, annotate,
  relabel, improve model, 主动学习, 数据标注, 模型改进.
---

# yolo-rural:active-learning — Collect → Annotate → Retrain → Deploy

## Step 1: Gather Inputs

Ask if not provided (one at a time):

1. **OOD queue directory** — where are low-confidence edge samples logged?
   (default: `./ood_queue/`)
2. **Confidence threshold** — samples below this score need review (default: 0.25)
3. **Annotation mode**:
   - `label-studio` — web GUI, good for teams
   - `cli` — command-line SAM annotation, good for solo use

## Step 2: Analyze OOD Queue

```python
python - <<'EOF'
import json, glob
from collections import Counter
from pathlib import Path

ood_dir = Path("./ood_queue")
meta_files = sorted(ood_dir.glob("*.json"))
img_files = list(ood_dir.glob("*.jpg")) + list(ood_dir.glob("*.png"))

print(f"OOD queue: {len(img_files)} images, {len(meta_files)} metadata files")

if not meta_files:
    print("Queue is empty — no samples to review yet.")
    print("Run the edge inference service and wait for low-confidence detections to accumulate.")
else:
    class_counts = Counter()
    conf_values = []
    for mf in meta_files:
        with open(mf) as f:
            data = json.load(f)
        for det in data.get("detections", []):
            class_counts[det.get("class", "unknown")] += 1
            conf_values.append(det.get("confidence", 0))
    avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0
    print(f"Average confidence: {avg_conf:.3f}")
    print("Class distribution in OOD queue:")
    for cls, cnt in class_counts.most_common():
        print(f"  {cls}: {cnt} detections")
EOF
```

If queue is empty: stop and tell the user to run the edge service longer before returning to this step.

## Step 3: Annotate Samples

### Option A: Label Studio (web GUI)

```bash
# Install if needed:
pip install label-studio

# Start Label Studio:
label-studio start --port 8080 &
echo "Label Studio running at http://localhost:8080"
echo ""
echo "Instructions:"
echo "  1. Create new project → Object Detection (Bounding Box)"
echo "  2. Import from local path: $(pwd)/ood_queue/"
echo "  3. Label each image with bounding boxes"
echo "  4. When done: Export → YOLO format → download to $(pwd)/ood_queue/labels/"
```

Wait for user to confirm annotation is complete before proceeding to Step 4.

### Option B: CLI with SAM (solo use)

Install SAM if needed:
```bash
pip install segment-anything
# Download ViT-B checkpoint (lighter and faster):
wget -q -O models/sam_vit_b.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Run CLI annotator:
```python
python - <<'PYEOF'
"""
Point-prompt SAM annotator.
Click on object → SAM generates mask → converts to YOLO bbox.
Keys: 's' = save labels, 'n' = next image, 'r' = redo, 'q' = quit.
"""
import sys, json, os
import numpy as np
import cv2
from pathlib import Path

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:
    print("ERROR: segment-anything not installed. Run: pip install segment-anything")
    sys.exit(1)

SAM_CKPT = "models/sam_vit_b.pth"
OOD_DIR = Path("./ood_queue")
CLASS_ID = int(input("Enter class ID to annotate (0-indexed): ") or "0")

sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
predictor = SamPredictor(sam)

images = sorted(OOD_DIR.glob("*.jpg")) + sorted(OOD_DIR.glob("*.png"))
images = [p for p in images if not p.with_suffix(".txt").exists()]
print(f"Found {len(images)} unlabeled images.")

for img_path in images:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    h, w = img_bgr.shape[:2]
    annotations = []
    display = img_bgr.copy()

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        masks, _, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        mask = masks[0]
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return
        x1, y1 = xs.min() / w, ys.min() / h
        x2, y2 = xs.max() / w, ys.max() / h
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = x2 - x1
        bh = y2 - y1
        annotations.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        overlay = display.copy()
        overlay[mask] = [0, 255, 0]
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
        cv2.imshow("SAM annotate", display)

    cv2.imshow("SAM annotate", display)
    cv2.setMouseCallback("SAM annotate", on_click)
    print(f"Annotating: {img_path.name} | 's'=save 'r'=redo 'n'=skip 'q'=quit")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s') and annotations:
            label_path = img_path.with_suffix(".txt")
            label_path.write_text('\n'.join(annotations))
            print(f"  Saved {len(annotations)} boxes → {label_path.name}")
            break
        elif key == ord('r'):
            annotations.clear()
            display = img_bgr.copy()
            cv2.imshow("SAM annotate", display)
        elif key == ord('n'):
            print(f"  Skipped: {img_path.name}")
            break
        elif key == ord('q'):
            print("Annotation session ended.")
            cv2.destroyAllWindows()
            sys.exit(0)

cv2.destroyAllWindows()
print("All images annotated.")
PYEOF
```

## Step 4: Merge New Annotations Into Dataset

```python
python - <<'EOF'
import shutil
from pathlib import Path
from datetime import datetime

ood_dir = Path("./ood_queue")
train_dir = Path("./data/train")
version_dir = Path("./data/versions") / datetime.now().strftime("%Y%m%d_%H%M%S")
version_dir.mkdir(parents=True, exist_ok=True)

# Snapshot current training set
for f in train_dir.iterdir():
    shutil.copy(f, version_dir / f.name)
print(f"Snapshot saved to: {version_dir}")

# Move labeled samples to train/
moved = 0
for img in list(ood_dir.glob("*.jpg")) + list(ood_dir.glob("*.png")):
    label = img.with_suffix(".txt")
    if label.exists():
        shutil.move(str(img), str(train_dir / img.name))
        shutil.move(str(label), str(train_dir / label.name))
        moved += 1

# Log to changelog
changelog = Path("data/versions/changelog.md")
with changelog.open("a") as f:
    f.write(f"\n## {datetime.now().isoformat()}\n")
    f.write(f"Added {moved} new samples from active learning queue.\n")

print(f"Merged {moved} new samples into data/train/")
print(f"Changelog updated: {changelog}")
EOF
```

## Step 5: Trigger Incremental Finetune

```bash
# Finetune from best checkpoint (freeze backbone for small increments)
PREV_BEST=$(find runs/ -name "best.pt" | sort | tail -1)
echo "Finetuning from: $PREV_BEST"

yolo train \
  model=$PREV_BEST \
  data=./data/data.yaml \
  epochs=30 \
  freeze=10 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=./runs \
  name=<scenario>_al_$(date +%Y%m%d)
```

Note: `freeze=10` freezes the first 10 backbone layers. Switch to `freeze=0` (full finetune) if total newly added images exceed 1000.

## Step 6: Compare Old vs New Model

```python
python - <<'EOF'
import subprocess
from pathlib import Path

def val_map50(model_path):
    r = subprocess.run(
        ["yolo", "val", f"model={model_path}", "data=./data/data.yaml",
         "split=val", "verbose=False", "plots=False"],
        capture_output=True, text=True
    )
    for line in r.stdout.splitlines():
        parts = line.split()
        # YOLO val output: "all  N  N  precision  recall  mAP50  mAP50-95"
        if parts and parts[0] == 'all' and len(parts) >= 7:
            try:
                return float(parts[5]), float(parts[6])
            except ValueError:
                pass
    return None, None

runs = sorted(Path("runs").glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime)
if len(runs) < 2:
    print("Need at least 2 checkpoints to compare. Skipping comparison.")
else:
    old_path = str(runs[-2])
    new_path = str(runs[-1])
    old_map50, old_map = val_map50(old_path)
    new_map50, new_map = val_map50(new_path)
    if old_map50 is not None and new_map50 is not None:
        print(f"Old model ({Path(old_path).parent.parent.name}): mAP50={old_map50:.3f}  mAP50-95={old_map:.3f}")
        print(f"New model ({Path(new_path).parent.parent.name}): mAP50={new_map50:.3f}  mAP50-95={new_map:.3f}")
        delta = new_map50 - old_map50
        print(f"Delta: {delta:+.3f}")
        if delta >= 0:
            print("IMPROVED — recommend deploying new model (run yolo-rural:deploy)")
        else:
            print("REGRESSED — do not deploy. Run yolo-rural:diagnose to investigate.")
    else:
        print("Could not parse metrics from validation output. Check runs/ manually.")
EOF
```

## Step 7: Report to User

```
Active learning cycle complete!
  New samples merged:  <N>
  Old mAP50:           <value>
  New mAP50:           <value>
  Delta:               <+/- value>

Dataset changelog:  data/versions/changelog.md
New checkpoint:     runs/<scenario>_al_<date>/weights/best.pt

Next:
  If improved → yolo-rural:deploy
  If regressed → yolo-rural:diagnose
```

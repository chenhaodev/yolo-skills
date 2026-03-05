---
name: yolo-rural:diagnose
description: >
  Debug and analyze rural CV model performance. Use when user wants to understand
  why a model is failing, view metrics, inspect confusion matrix, find worst-performing
  samples, or get improvement recommendations. Triggered by: diagnose, debug, performance,
  metrics, confusion matrix, why is it failing, low accuracy, 诊断, 调试, 性能分析.
---

# yolo-rural:diagnose — Performance Diagnosis

## Step 1: Gather Inputs

Ask if not provided (one at a time):

1. **Checkpoint path** — which model to diagnose?
   (default: find latest `runs/**/weights/best.pt`)
2. **Test data path** — where is the test/val set? (default: `./data/test/` or `./data/val/`)
3. **Scenario** — which scenario? Needed for scenario-specific failure checks.

## Step 2: Run Full Validation

```bash
# Use conf=0.001 to get the full PR curve, not just detections above threshold
CKPT=<checkpoint>
DATA=<data-path>/data.yaml

yolo val \
  model=$CKPT \
  data=$DATA \
  split=<test-or-val> \
  save_json=True \
  plots=True \
  conf=0.001 \
  iou=0.6 \
  project=./diagnostics \
  name=diag_$(date +%Y%m%d)
```

Note: This produces confusion_matrix.png, PR_curve.png, results.csv, and predictions.json in `diagnostics/diag_<date>/`.

## Step 3: Parse and Display Overall Metrics

```python
python - <<'EOF'
import csv, os
from pathlib import Path

results_files = sorted(Path("diagnostics").glob("**/results.csv"), key=os.path.getmtime)
if not results_files:
    # Fallback: check runs/ directory
    results_files = sorted(Path("runs").glob("**/results.csv"), key=os.path.getmtime)
if not results_files:
    print("No results.csv found. Ensure yolo val ran successfully.")
else:
    with open(results_files[-1]) as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    metrics = {
        "mAP50":      last.get("metrics/mAP50(B)", "N/A"),
        "mAP50-95":   last.get("metrics/mAP50-95(B)", "N/A"),
        "Precision":  last.get("metrics/precision(B)", "N/A"),
        "Recall":     last.get("metrics/recall(B)", "N/A"),
        "Box Loss":   last.get("val/box_loss", "N/A"),
        "Class Loss": last.get("val/cls_loss", "N/A"),
    }
    print("=== Validation Metrics ===")
    for label, val in metrics.items():
        try:
            print(f"  {label:12s}: {float(val):.4f}")
        except (ValueError, TypeError):
            print(f"  {label:12s}: {val}")
EOF
```

## Step 4: Per-Class Analysis

```python
python - <<'EOF'
import json
from pathlib import Path
from collections import defaultdict

pred_files = sorted(Path("diagnostics").glob("**/predictions.json"), key=lambda p: p.stat().st_mtime)
if not pred_files:
    pred_files = sorted(Path("runs").glob("**/predictions.json"), key=lambda p: p.stat().st_mtime)

if not pred_files:
    print("No predictions.json found. Re-run with save_json=True.")
else:
    with open(pred_files[-1]) as f:
        preds = json.load(f)
    if not preds:
        print("predictions.json is empty — model may not have detected anything above conf=0.001.")
    class_confs = defaultdict(list)
    for p in preds:
        cls_id = p.get("category_id", "?")
        class_confs[cls_id].append(p.get("score", 0))
    print("=== Per-class confidence ===")
    for cls_id in sorted(class_confs):
        confs = class_confs[cls_id]
        avg = sum(confs) / len(confs)
        low = sum(1 for c in confs if c < 0.5)
        print(f"  Class {cls_id}: n={len(confs):4d}  avg_conf={avg:.3f}  low_conf_ratio={low/len(confs):.2f}")
EOF
```

## Step 5: Find Worst-Performing Samples

```python
python - <<'EOF'
import json
from pathlib import Path

pred_files = sorted(Path("diagnostics").glob("**/predictions.json"), key=lambda p: p.stat().st_mtime)
if not pred_files:
    pred_files = sorted(Path("runs").glob("**/predictions.json"), key=lambda p: p.stat().st_mtime)

if not pred_files:
    print("No predictions.json found.")
else:
    with open(pred_files[-1]) as f:
        preds = json.load(f)
    worst = sorted(preds, key=lambda x: x.get("score", 1.0))[:10]
    print("=== 10 Worst Predictions (lowest confidence) ===")
    for p in worst:
        print(f"  image_id={p.get('image_id')}  class={p.get('category_id')}  conf={p.get('score', 0):.3f}")
    print(f"\nPlots saved at:")
    for pf in pred_files[-1].parent.parent.glob("*.png"):
        print(f"  {pf}")
EOF
```

## Step 6: Scenario-Specific Checks

After reviewing metrics, apply these scenario-specific checks:

**livestock-stray (牛马占道):**
- Common failure: animals at night or in rain (low contrast) — check val images for dark scenes
- Verify zone polygon coordinates in `deployment/edge_service.py` if alerts aren't triggering
- If Recall < 0.7: add more diverse background images (roads without animals) to reduce false negatives

**public-disorder (打架斗殴):**
- Common failure: normal crowd activity detected as fighting — need more negative examples
- Verify pose keypoints are being detected (run `yolo predict model=<ckpt> source=<image> show=True`)
- Temporal consistency: single frames may miss context; consider sliding window over video frames

**fire-safety (明火识别):**
- Recall is critical — if Recall < 0.85, lower production threshold: `conf=0.15`
- Common false positives: sunlight, reflections, red lights — add these as negative training samples
- Test with nighttime images and infrared camera outputs if applicable

**illegal-fishing (垂钓检测):**
- Time-of-day failures: check if model was trained on balanced day/night images
- Fishing rod is small object — increase imgsz to 1280 if currently at 640
- Verify `require_both=True` logic in alerts.py (person AND rod must co-occur)

**disease-recognition (病害识别):**
- Check confusion between similar disease classes in the confusion matrix
- Image quality is critical: ensure training images match field conditions (lighting, angle, distance)
- Consider ensemble: combine YOLOv8-cls with MMPretrain for higher accuracy

## Step 7: Generate Recommendations

Based on overall mAP50, apply these rules and report all that apply:

```
mAP50 < 0.3  → CRITICAL: barely learning
  Checks: (1) Verify label format is correct YOLO (not COCO). (2) Check for class
  index mismatch (class IDs in labels must match data.yaml nc). (3) Inspect 5
  random training images with labels: yolo detect predict model=<ckpt> source=<img> show=True

mAP50 0.3–0.5 → LOW: significant gaps
  Likely causes: insufficient data or high class imbalance.
  Actions: (1) Run yolo-rural:active-learning to add more diverse samples.
  (2) Check class balance — dominant classes suppress minority class learning.

mAP50 0.5–0.7 → MODERATE: usable but improvable
  Actions: (1) Add 1-2 active learning cycles. (2) Try larger model (yolov8s→yolov8m).
  (3) Enable test-time augmentation: yolo val ... augment=True

mAP50 > 0.7 → GOOD: focus on weakest class and edge cases
  Actions: (1) Identify lowest-recall class from per-class analysis.
  (2) Collect targeted samples for that class via active learning.

Recall < 0.5 on fire-safety or public-disorder → SAFETY WARNING
  Lower conf threshold to 0.15 in production (accept more false positives
  to avoid missing real events). Alert the user explicitly.

Precision < 0.4 → Too many false positives
  Actions: (1) Add negative examples (background / non-target images) to training.
  (2) Raise conf threshold. (3) Add zone/time filtering in alerts.py.
```

## Step 8: Report to User

```
=== Diagnosis Report ===
Scenario:    <scenario>
Checkpoint:  <path>
Test split:  <test|val>

Metrics:
  mAP50:     <value>  [<poor/moderate/good/excellent>]
  mAP50-95:  <value>
  Precision: <value>
  Recall:    <value>

Key Findings:
  <list each finding from Steps 4-7>

Recommendations:
  1. <top action>
  2. <second action>
  3. <third action if needed>

Plots: diagnostics/diag_<date>/
  - confusion_matrix.png
  - PR_curve.png
  - results.png

Next steps: yolo-rural:active-learning (more data) | yolo-rural:train (retrain)
```

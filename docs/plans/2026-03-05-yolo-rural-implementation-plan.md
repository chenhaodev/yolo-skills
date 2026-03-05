# YOLO Rural Skill Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build 6 Claude Code SKILL.md files forming the `yolo-rural` skill suite for rural/community CV pipelines (livestock, fire, disorder, fishing, disease).

**Architecture:** Modular skill family under `~/.claude/skills/yolo-rural/`. Root skill routes to 5 focused sub-skills. Each skill is framework-agnostic (auto-detects Ultralytics YOLO or OpenMMLab).

**Tech Stack:** Ultralytics YOLO (YOLOv8/YOLO11), OpenMMLab (MMDetection/MMClassification), SAM (Segment Anything), Label Studio, ONNX, TensorRT, Docker

---

## Reference

Design doc: `docs/plans/2026-03-05-yolo-rural-skill-suite-design.md`

Skill install target: `~/.claude/skills/yolo-rural/`

5 scenarios:
- `livestock-stray` — 牛马占道 (cattle/horses on roads)
- `public-disorder` — 打架斗殴 (fighting detection)
- `fire-safety` — 明火识别 (open flame/smoke)
- `illegal-fishing` — 垂钓检测 (fishing at waterways)
- `disease-recognition` — 病害识别 (livestock disease)

---

## Task 1: Create Directory Structure

**Files:**
- Create: `~/.claude/skills/yolo-rural/` (directory tree)

**Step 1: Create all skill directories**

```bash
mkdir -p ~/.claude/skills/yolo-rural/scaffold
mkdir -p ~/.claude/skills/yolo-rural/train
mkdir -p ~/.claude/skills/yolo-rural/deploy
mkdir -p ~/.claude/skills/yolo-rural/active-learning
mkdir -p ~/.claude/skills/yolo-rural/diagnose
```

**Step 2: Verify structure**

```bash
ls ~/.claude/skills/yolo-rural/
```
Expected output:
```
active-learning  deploy  diagnose  scaffold  train
```

**Step 3: Commit**

```bash
cd /Users/chenhao/MyOpenCode/yolo-skills
git add -A
git commit -m "chore: scaffold yolo-rural skill directory structure"
```

---

## Task 2: Write Root Router Skill

**Files:**
- Create: `~/.claude/skills/yolo-rural/SKILL.md`

**Step 1: Write the file**

Create `~/.claude/skills/yolo-rural/SKILL.md` with this exact content:

````markdown
---
name: yolo-rural
description: >
  Rural/community computer vision pipeline manager. Use when the user mentions
  livestock detection, cattle/horse straying, fire detection, fighting/disorder,
  fishing detection, livestock disease, or any rural/farm/community monitoring scenario.
  Also triggered by: yolo-rural, 牛马占道, 打架斗殴, 明火识别, 垂钓检测, 病害识别.
---

# YOLO Rural — Scenario Router

You are helping build or manage a rural/community computer vision pipeline.

## Step 1: Detect Installed Frameworks

Run these checks silently:

```bash
python -c "import ultralytics; print('YOLO:', ultralytics.__version__)" 2>/dev/null || echo "YOLO: not installed"
python -c "import mmdet; print('MMDet:', mmdet.__version__)" 2>/dev/null || echo "MMDet: not installed"
python -c "import mmcls; print('MMCls:', mmcls.__version__)" 2>/dev/null || echo "MMCls: not installed"
python -c "import segment_anything; print('SAM: installed')" 2>/dev/null || echo "SAM: not installed"
```

Report to user: "Detected frameworks: [list what's installed]"

## Step 2: Show Scenario Status

Check for existing project directories and show:

```
Scenario              | Status        | Recommended Model
--------------------- | ------------- | ------------------
livestock-stray       | [configured?] | YOLOv8m / YOLO11m
public-disorder       | [configured?] | YOLOv8-pose
fire-safety           | [configured?] | YOLOv8n-seg
illegal-fishing       | [configured?] | YOLOv8s
disease-recognition   | [configured?] | MMCls / YOLOv8-cls
```

## Step 3: Route to Sub-skill

Based on user intent, invoke the appropriate sub-skill:

| User intent | Sub-skill to invoke |
|------------|---------------------|
| "set up", "create project", "scaffold", "new" | `yolo-rural:scaffold` |
| "train", "finetune", "retrain", "fine-tune" | `yolo-rural:train` |
| "deploy", "export", "edge", "jetson", "docker" | `yolo-rural:deploy` |
| "active learning", "new data", "annotate", "relabel" | `yolo-rural:active-learning` |
| "debug", "diagnose", "performance", "metrics", "why is it failing" | `yolo-rural:diagnose` |

If intent is unclear, ask:
> "What would you like to do?
> 1. Set up a new project (scaffold)
> 2. Train / finetune a model (train)
> 3. Deploy to edge or cloud (deploy)
> 4. Add new data and retrain (active-learning)
> 5. Debug model performance (diagnose)"

## Framework Selection Rules

When both YOLO and OpenMMLab are installed:
- **Real-time edge scenarios** (livestock-stray, fire-safety, illegal-fishing, public-disorder): use YOLO
- **Disease recognition** (disease-recognition): prefer MMClassification for accuracy; fall back to YOLOv8-cls if MMCls not installed

When neither is installed:
- For edge/real-time scenarios: recommend `pip install ultralytics`
- For disease recognition: recommend `pip install -U openmim && mim install mmcls` or `pip install ultralytics`
````

**Step 2: Verify file exists**

```bash
ls -la ~/.claude/skills/yolo-rural/SKILL.md
```
Expected: file present, non-zero size.

**Step 3: Commit**

```bash
cd /Users/chenhao/MyOpenCode/yolo-skills
cp ~/.claude/skills/yolo-rural/SKILL.md skills/yolo-rural/SKILL.md 2>/dev/null || true
git add -A
git commit -m "feat: add yolo-rural root router skill"
```

---

## Task 3: Write Scaffold Skill

**Files:**
- Create: `~/.claude/skills/yolo-rural/scaffold/SKILL.md`

**Step 1: Write the file**

Create `~/.claude/skills/yolo-rural/scaffold/SKILL.md`:

````markdown
---
name: yolo-rural:scaffold
description: >
  Scaffold a new rural CV project for a specific scenario. Use when user wants
  to create a new project, set up directory structure, download pretrained weights,
  or initialize a detection pipeline for livestock, fire, disorder, fishing, or disease.
---

# yolo-rural:scaffold — Project Scaffolding

## Step 1: Gather Inputs

If not provided, ask these questions **one at a time**:

1. **Scenario** — Which scenario?
   - `livestock-stray` (牛马占道)
   - `public-disorder` (打架斗殴)
   - `fire-safety` (明火识别)
   - `illegal-fishing` (垂钓检测)
   - `disease-recognition` (病害识别)

2. **Project directory** — Where to create the project? (default: `./<scenario>-project`)

3. **Edge target** — What hardware will run inference?
   - `jetson-orin`, `jetson-nano`, `raspberry-pi`, `server`, `cloud-only`

4. **Framework** — Auto-detect (recommended), `yolo-only`, or `mmlab-only`?

## Step 2: Detect Framework

```bash
python -c "import ultralytics; print('yolo')" 2>/dev/null
python -c "import mmdet; print('mmdet')" 2>/dev/null
python -c "import mmcls; print('mmcls')" 2>/dev/null
```

If nothing installed:
- For real-time scenarios: run `pip install ultralytics`
- For disease-recognition: run `pip install ultralytics` (YOLOv8-cls as fallback) or `pip install -U openmim && mim install mmcls`

## Step 3: Create Project Structure

```bash
PROJECT_DIR="<user-specified-or-default>"
mkdir -p "$PROJECT_DIR"/{data/{raw,train,val,test,versions},configs,scripts,models/pretrained,deployment,notebooks}
```

## Step 4: Generate data.yaml (YOLO format)

For each scenario, use these class definitions:

**livestock-stray:**
```yaml
# data/data.yaml
path: ./data
train: train
val: val
test: test
nc: 2
names: ['cattle', 'horse']
```

**public-disorder:**
```yaml
path: ./data
train: train
val: val
test: test
nc: 1
names: ['fighting']
# Note: uses pose model — keypoint anomaly classifier built separately
```

**fire-safety:**
```yaml
path: ./data
train: train
val: val
test: test
nc: 2
names: ['flame', 'smoke']
```

**illegal-fishing:**
```yaml
path: ./data
train: train
val: val
test: test
nc: 2
names: ['person', 'fishing-rod']
```

**disease-recognition:**
```yaml
path: ./data
train: train
val: val
test: test
nc: 5
names: ['healthy', 'skin-lesion', 'respiratory', 'digestive', 'parasitic']
# Adjust class names to actual diseases in your livestock population
```

## Step 5: Generate YOLO Training Config

Create `configs/yolo_<scenario>.yaml`:

```yaml
# Model selection by scenario
# livestock-stray:     yolov8m.pt
# public-disorder:     yolov8m-pose.pt
# fire-safety:         yolov8n-seg.pt
# illegal-fishing:     yolov8s.pt
# disease-recognition: yolov8s-cls.pt

model: <model-from-table-above>
data: ./data/data.yaml
epochs: 100
imgsz: 640
batch: 16
device: 0          # GPU index; use 'cpu' if no GPU
workers: 8
project: ./runs
name: <scenario>_v1
save: true
val: true
plots: true

# OOD: log low-confidence predictions for active learning
conf: 0.25         # detection threshold
iou: 0.45

# Augmentation (adjust per scenario)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
```

## Step 6: Generate Shell Scripts

**scripts/train.sh:**
```bash
#!/bin/bash
# Usage: ./scripts/train.sh [--resume]
set -e
SCENARIO="<scenario>"
CONFIG="configs/yolo_${SCENARIO}.yaml"
if [ "$1" = "--resume" ]; then
  yolo train cfg=$CONFIG resume=True
else
  yolo train cfg=$CONFIG
fi
```

**scripts/export.sh:**
```bash
#!/bin/bash
# Usage: ./scripts/export.sh <checkpoint.pt> <format: onnx|engine>
set -e
WEIGHTS=${1:-"runs/<scenario>_v1/weights/best.pt"}
FORMAT=${2:-"onnx"}
yolo export model=$WEIGHTS format=$FORMAT
echo "Exported to: ${WEIGHTS%.pt}.${FORMAT}"
```

**scripts/infer.sh:**
```bash
#!/bin/bash
# Usage: ./scripts/infer.sh <source: image/video/rtsp_url>
set -e
WEIGHTS="runs/<scenario>_v1/weights/best.pt"
SOURCE=${1:-"0"}  # 0 = webcam
yolo predict model=$WEIGHTS source=$SOURCE conf=0.25 save=True save_txt=True
```

**scripts/annotate.sh:**
```bash
#!/bin/bash
# Launch Label Studio for annotation
# Requires: pip install label-studio
set -e
label-studio start --port 8080 &
echo "Label Studio running at http://localhost:8080"
echo "Import images from: $(pwd)/data/raw/"
```

Make scripts executable:
```bash
chmod +x scripts/*.sh
```

## Step 7: Generate Deployment Files

**deployment/Dockerfile:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN pip install ultralytics fastapi uvicorn python-multipart
COPY models/pretrained/ models/pretrained/
COPY deployment/edge_service.py .
EXPOSE 8000
CMD ["uvicorn", "edge_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

**deployment/edge_service.py:**
```python
"""Minimal inference API for rural CV scenarios."""
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()
model = YOLO("models/pretrained/best.pt")

OOD_CONF_THRESHOLD = 0.25  # below this → OOD queue

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read()))
    results = model(img)
    detections = []
    ood_samples = []
    for r in results:
        for box in r.boxes:
            det = {
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist(),
            }
            if float(box.conf) < OOD_CONF_THRESHOLD:
                ood_samples.append(det)
            else:
                detections.append(det)
    return {"detections": detections, "ood_queue": ood_samples}

@app.get("/health")
async def health():
    return {"status": "ok"}
```

**deployment/alerts.py:**
```python
"""Scenario-specific alert rules."""
ALERT_RULES = {
    "livestock-stray": {
        "classes": ["cattle", "horse"],
        "zone_crossing": True,   # trigger alert when animal enters road zone
        "cooldown_seconds": 30,
    },
    "public-disorder": {
        "classes": ["fighting"],
        "min_persons": 2,
        "cooldown_seconds": 10,
    },
    "fire-safety": {
        "classes": ["flame", "smoke"],
        "zone_crossing": False,  # alert on any detection
        "cooldown_seconds": 5,
    },
    "illegal-fishing": {
        "classes": ["person", "fishing-rod"],
        "require_both": True,    # only alert if person + rod both detected
        "time_window": "06:00-20:00",
        "cooldown_seconds": 60,
    },
    "disease-recognition": {
        "classes": ["skin-lesion", "respiratory", "digestive", "parasitic"],
        "batch_mode": True,      # no real-time alerting; batch report
    },
}
```

## Step 8: Download Pretrained Weights

```bash
cd models/pretrained/
# Download appropriate base model
python -c "
from ultralytics import YOLO
scenario_models = {
    'livestock-stray': 'yolov8m.pt',
    'public-disorder': 'yolov8m-pose.pt',
    'fire-safety': 'yolov8n-seg.pt',
    'illegal-fishing': 'yolov8s.pt',
    'disease-recognition': 'yolov8s-cls.pt',
}
model_name = scenario_models.get('<scenario>', 'yolov8s.pt')
YOLO(model_name)  # downloads automatically
print(f'Downloaded: {model_name}')
"
```

## Step 9: Generate README.md

Create `README.md` with scenario-specific content:

```markdown
# <Scenario Name> — Rural CV Project

## Scenario
<Chinese name> | <English description>

## Quick Start
\`\`\`bash
# 1. Add images to data/raw/
# 2. Annotate with Label Studio
./scripts/annotate.sh

# 3. Train
./scripts/train.sh

# 4. Export for edge
./scripts/export.sh runs/<scenario>_v1/weights/best.pt onnx

# 5. Run inference
./scripts/infer.sh /path/to/video.mp4
\`\`\`

## Data Requirements
- Minimum: 500 images (this scenario)
- Format: YOLO txt (images in data/train/, labels in data/train/ as .txt)
- Use active learning to grow dataset iteratively

## Next Steps
- Add more data: use `yolo-rural:active-learning`
- Deploy to edge: use `yolo-rural:deploy`
- Debug performance: use `yolo-rural:diagnose`
```

## Step 10: Confirm to User

Report:
```
Project scaffolded at: <project_dir>/
  data/          - Add your images here (train/val/test splits)
  configs/       - Training configuration
  scripts/       - train.sh, export.sh, infer.sh, annotate.sh
  models/        - Pretrained weights downloaded
  deployment/    - Edge service + Docker + alerts

Next steps:
  1. Add labeled images to data/train/ and data/val/
  2. Run: ./scripts/train.sh
  3. Or collect data first: ./scripts/annotate.sh (Label Studio on :8080)
```
````

**Step 2: Verify**

```bash
wc -l ~/.claude/skills/yolo-rural/scaffold/SKILL.md
```
Expected: under 400 lines.

**Step 3: Commit**

```bash
cd /Users/chenhao/MyOpenCode/yolo-skills
mkdir -p skills/yolo-rural/scaffold
cp ~/.claude/skills/yolo-rural/scaffold/SKILL.md skills/yolo-rural/scaffold/SKILL.md
git add -A
git commit -m "feat: add yolo-rural:scaffold skill"
```

---

## Task 4: Write Train Skill

**Files:**
- Create: `~/.claude/skills/yolo-rural/train/SKILL.md`

**Step 1: Write the file**

Create `~/.claude/skills/yolo-rural/train/SKILL.md`:

````markdown
---
name: yolo-rural:train
description: >
  Train or finetune a YOLO/OpenMMLab model for a rural CV scenario. Use when user
  wants to train, finetune, retrain, or fine-tune a model. Works with Ultralytics
  YOLO and OpenMMLab frameworks. Validates data, runs training, evaluates results.
---

# yolo-rural:train — Training & Finetuning

## Step 1: Gather Inputs

Ask if not provided (one at a time):

1. **Data path** — where is your dataset? (default: `./data`)
2. **Scenario** — which scenario? (livestock-stray / public-disorder / fire-safety / illegal-fishing / disease-recognition)
3. **Epochs** — how many epochs? (default: 100 for new train, 30 for finetune)
4. **Device** — GPU index or 'cpu'? (default: 0)
5. **Resume?** — resuming from a checkpoint? If yes, which checkpoint path?

## Step 2: Detect Framework

```bash
python -c "import ultralytics; print('yolo')" 2>/dev/null && echo "framework=yolo"
python -c "import mmcls; print('mmcls')" 2>/dev/null && echo "framework=mmcls"
```

## Step 3: Validate Dataset

```bash
python - <<'EOF'
import os, glob

data_path = "<user-data-path>"
splits = ['train', 'val']
for split in splits:
    img_dir = os.path.join(data_path, split)
    if not os.path.exists(img_dir):
        print(f"ERROR: missing {img_dir}")
        continue
    imgs = glob.glob(f"{img_dir}/*.jpg") + glob.glob(f"{img_dir}/*.png")
    txts = glob.glob(f"{img_dir}/*.txt")
    print(f"{split}: {len(imgs)} images, {len(txts)} labels")
    if len(imgs) == 0:
        print(f"WARNING: no images in {img_dir}")
    if len(txts) == 0 and "<scenario>" != "disease-recognition":
        print(f"WARNING: no label files in {img_dir}")
EOF
```

If validation fails, stop and guide user:
- Missing images: "Add images to `data/train/` and `data/val/`. Run `./scripts/annotate.sh` to label them."
- Missing labels: "Label your images first. Run `./scripts/annotate.sh` to launch Label Studio."
- Format mismatch: proceed to Step 4 (auto-convert).

## Step 4: Auto-Convert Format If Needed

If labels are in COCO JSON format, convert to YOLO:

```python
# Run only if data is in COCO format
python - <<'EOF'
import json, os
from pathlib import Path

def coco_to_yolo(coco_json_path, output_dir):
    with open(coco_json_path) as f:
        coco = json.load(f)
    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    id_to_size = {img['id']: (img['width'], img['height']) for img in coco['images']}
    labels = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        w, h = id_to_size[img_id]
        x, y, bw, bh = ann['bbox']
        cx = (x + bw/2) / w
        cy = (y + bh/2) / h
        nw = bw / w
        nh = bh / h
        cls = ann['category_id'] - 1
        stem = Path(id_to_file[img_id]).stem
        labels.setdefault(stem, []).append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    os.makedirs(output_dir, exist_ok=True)
    for stem, lines in labels.items():
        with open(f"{output_dir}/{stem}.txt", 'w') as f:
            f.write('\n'.join(lines))
    print(f"Converted {len(labels)} label files to {output_dir}")

# coco_to_yolo("data/annotations/instances_train.json", "data/train")
EOF
```

## Step 5: Run Training

### YOLO (Ultralytics):

```bash
# Standard detection / segmentation / pose / classification
yolo train \
  model=<model-from-scenario> \
  data=<data-path>/data.yaml \
  epochs=<epochs> \
  imgsz=640 \
  batch=16 \
  device=<device> \
  project=./runs \
  name=<scenario>_v<next-version> \
  save=True \
  plots=True \
  conf=0.25 \
  iou=0.45
```

### OpenMMLab (disease-recognition with MMClassification):

```bash
python $(python -c "import mmcls; import os; print(os.path.dirname(mmcls.__file__))")/tools/train.py \
  configs/mm_disease_recognition.py \
  --work-dir ./runs/disease_v<version> \
  --gpu-id <device>
```

Generate `configs/mm_disease_recognition.py`:
```python
_base_ = ['mmcls::resnet/resnet50_8xb32_in1k.py']
model = dict(head=dict(num_classes=5))  # adjust to actual class count
data_root = './data/'
train_dataloader = dict(dataset=dict(data_root=data_root, ann_file='train.txt'))
val_dataloader = dict(dataset=dict(data_root=data_root, ann_file='val.txt'))
max_epochs = 50
```

## Step 6: Monitor Training

While training runs, tell user:
- YOLO outputs live to terminal — watch for `mAP50`, `mAP50-95` metrics
- Checkpoints saved to `runs/<scenario>_v<N>/weights/best.pt` and `last.pt`
- TensorBoard available: `tensorboard --logdir runs/`

## Step 7: Post-Training Evaluation

After training completes:

```bash
# YOLO validation
yolo val model=runs/<scenario>_v<N>/weights/best.pt data=<data-path>/data.yaml

# Extract key metrics
python - <<'EOF'
import json
from pathlib import Path
results_file = sorted(Path('runs').glob('**/results.csv'))[-1]
import csv
with open(results_file) as f:
    rows = list(csv.DictReader(f))
last = rows[-1]
print(f"mAP50: {float(last.get('metrics/mAP50(B)', 0)):.3f}")
print(f"mAP50-95: {float(last.get('metrics/mAP50-95(B)', 0)):.3f}")
print(f"Precision: {float(last.get('metrics/precision(B)', 0)):.3f}")
print(f"Recall: {float(last.get('metrics/recall(B)', 0)):.3f}")
EOF
```

## Step 8: Report to User

```
Training complete!
  Checkpoint: runs/<scenario>_v<N>/weights/best.pt
  mAP50:      <value>
  mAP50-95:   <value>
  Precision:  <value>
  Recall:     <value>

Next steps:
  - Deploy: yolo-rural:deploy
  - Debug low performance: yolo-rural:diagnose
  - Add more data: yolo-rural:active-learning
```

If mAP50 < 0.5, proactively suggest: "Performance is below 0.5 mAP50. Consider:
1. Adding more diverse training data (active learning)
2. Running `yolo-rural:diagnose` to identify failure modes
3. Increasing model size (e.g., yolov8m → yolov8l)"
````

**Step 2: Verify**

```bash
wc -l ~/.claude/skills/yolo-rural/train/SKILL.md
```
Expected: under 400 lines.

**Step 3: Commit**

```bash
cd /Users/chenhao/MyOpenCode/yolo-skills
mkdir -p skills/yolo-rural/train
cp ~/.claude/skills/yolo-rural/train/SKILL.md skills/yolo-rural/train/SKILL.md
git add -A
git commit -m "feat: add yolo-rural:train skill"
```

---

## Task 5: Write Deploy Skill

**Files:**
- Create: `~/.claude/skills/yolo-rural/deploy/SKILL.md`

**Step 1: Write the file**

Create `~/.claude/skills/yolo-rural/deploy/SKILL.md`:

````markdown
---
name: yolo-rural:deploy
description: >
  Deploy a trained YOLO/OpenMMLab model to edge devices or cloud. Use when user
  wants to export, deploy, run on Jetson, Raspberry Pi, Docker, or package for
  production. Handles ONNX export, TensorRT optimization, systemd service, Docker.
---

# yolo-rural:deploy — Edge & Cloud Deployment

## Step 1: Gather Inputs

Ask if not provided (one at a time):

1. **Checkpoint path** — which model? (default: `runs/<latest>/weights/best.pt`)
2. **Target platform** — where is this deploying?
   - `jetson-orin` — Jetson Orin (TensorRT + FP16)
   - `jetson-nano` — Jetson Nano (TensorRT + INT8)
   - `raspberry-pi` — RPi (ONNX Runtime, CPU)
   - `server` — On-premise GPU server (ONNX or PyTorch)
   - `cloud` — Cloud API (Docker + FastAPI)
3. **Precision** — `fp32`, `fp16`, or `int8`? (default: fp16 for Jetson, fp32 for others)
4. **Scenario** — needed for alert rules configuration

## Step 2: Export Model

### ONNX export (universal first step):
```bash
yolo export \
  model=<checkpoint> \
  format=onnx \
  imgsz=640 \
  half=<True if fp16 else False> \
  dynamic=False \
  simplify=True
```

### TensorRT export (Jetson only — run ON the Jetson device):
```bash
# Run this command ON the Jetson, not on training server
yolo export \
  model=<checkpoint> \
  format=engine \
  imgsz=640 \
  half=True \
  device=0
```

Note: TensorRT engines are device-specific. Build on the exact target device.

### INT8 calibration (Jetson Nano with limited VRAM):
```bash
yolo export \
  model=<checkpoint> \
  format=engine \
  imgsz=640 \
  int8=True \
  data=<data-path>/data.yaml \
  device=0
```

## Step 3: Generate Inference Wrapper

Create `deployment/edge_service.py` (update with scenario-specific logic):

```python
"""Edge inference service with zone logic, OOD flagging, and alerts."""
import os, json, time, logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, BackgroundTasks
from ultralytics import YOLO
import io
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="yolo-rural edge service")

SCENARIO = os.environ.get("SCENARIO", "<scenario>")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
OOD_THRESHOLD = float(os.environ.get("OOD_THRESHOLD", "0.25"))
OOD_LOG_DIR = Path(os.environ.get("OOD_LOG_DIR", "./ood_queue"))
OOD_LOG_DIR.mkdir(exist_ok=True)

model = YOLO(MODEL_PATH)

def log_ood_sample(image: Image.Image, detections: list):
    """Save low-confidence samples for active learning review."""
    ts = int(time.time())
    img_path = OOD_LOG_DIR / f"{ts}.jpg"
    meta_path = OOD_LOG_DIR / f"{ts}.json"
    image.save(img_path)
    with open(meta_path, 'w') as f:
        json.dump({"timestamp": ts, "detections": detections}, f)

@app.post("/predict")
async def predict(file: UploadFile, background_tasks: BackgroundTasks):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))
    results = model(img, conf=OOD_THRESHOLD)
    detections, ood = [], []
    for r in results:
        for box in r.boxes:
            det = {
                "class": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "bbox": [round(x, 1) for x in box.xyxy[0].tolist()],
            }
            if float(box.conf) < OOD_THRESHOLD:
                ood.append(det)
            else:
                detections.append(det)
    if ood:
        background_tasks.add_task(log_ood_sample, img, ood)
    return {"scenario": SCENARIO, "detections": detections, "ood_count": len(ood)}

@app.get("/health")
async def health():
    return {"status": "ok", "scenario": SCENARIO, "model": MODEL_PATH}
```

## Step 4: Platform-Specific Packaging

### Cloud / Server (Docker):

Update `deployment/Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir ultralytics fastapi uvicorn python-multipart pillow
COPY models/ models/
COPY deployment/edge_service.py .
COPY deployment/alerts.py .
ENV SCENARIO=<scenario>
ENV MODEL_PATH=models/best.pt
ENV OOD_THRESHOLD=0.25
EXPOSE 8000
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "edge_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t yolo-rural-<scenario>:latest -f deployment/Dockerfile .
docker run -d -p 8000:8000 --name rural-<scenario> yolo-rural-<scenario>:latest
```

### Edge — systemd service (Jetson/RPi):

Create `deployment/rural-cv.service`:
```ini
[Unit]
Description=Rural CV Inference Service — <scenario>
After=network.target

[Service]
Type=simple
User=<user>
WorkingDirectory=<project-path>
Environment="SCENARIO=<scenario>"
Environment="MODEL_PATH=<project-path>/models/best.pt"
Environment="OOD_LOG_DIR=<project-path>/ood_queue"
ExecStart=/usr/bin/python3 -m uvicorn deployment.edge_service:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Install:
```bash
sudo cp deployment/rural-cv.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rural-cv
sudo systemctl start rural-cv
sudo systemctl status rural-cv
```

## Step 5: Generate deploy.yaml (Ansible/SSH push)

```yaml
# deployment/deploy.yaml — push model update to edge devices
---
- hosts: edge_devices
  vars:
    project_path: /opt/rural-cv/<scenario>
    model_src: "{{ playbook_dir }}/../runs/{{ scenario }}_latest/weights/best.pt"
  tasks:
    - name: Copy new model weights
      copy:
        src: "{{ model_src }}"
        dest: "{{ project_path }}/models/best.pt"
        backup: yes
    - name: Restart inference service
      systemd:
        name: rural-cv
        state: restarted
```

Run: `ansible-playbook -i inventory.ini deployment/deploy.yaml`

## Step 6: Verify Deployment

```bash
# Test the running service
curl -X POST http://<device-ip>:8000/predict \
  -F "file=@data/val/sample.jpg" | python3 -m json.tool

# Check health
curl http://<device-ip>:8000/health
```

## Step 7: Report to User

```
Deployment complete!
  Format:    <onnx|engine|docker>
  Target:    <platform>
  Precision: <fp32|fp16|int8>

Edge service running at: http://<device-ip>:8000
  POST /predict  — run inference
  GET  /health   — health check

OOD samples logged to: <ood_log_dir>/
  Use yolo-rural:active-learning to review and retrain on these.
```
````

**Step 2: Commit**

```bash
cd /Users/chenhao/MyOpenCode/yolo-skills
mkdir -p skills/yolo-rural/deploy
cp ~/.claude/skills/yolo-rural/deploy/SKILL.md skills/yolo-rural/deploy/SKILL.md
git add -A
git commit -m "feat: add yolo-rural:deploy skill"
```

---

## Task 6: Write Active Learning Skill

**Files:**
- Create: `~/.claude/skills/yolo-rural/active-learning/SKILL.md`

**Step 1: Write the file**

Create `~/.claude/skills/yolo-rural/active-learning/SKILL.md`:

````markdown
---
name: yolo-rural:active-learning
description: >
  Active learning loop for rural CV models: collect low-confidence edge samples,
  annotate with SAM assistance, merge into dataset, retrain, compare, redeploy.
  Use when user wants to improve model with new data, retrain on recent failures,
  or run the annotation → train → deploy cycle.
---

# yolo-rural:active-learning — Collect → Annotate → Retrain → Deploy

## Step 1: Gather Inputs

Ask if not provided:
1. **OOD queue directory** — where are low-confidence samples logged? (default: `./ood_queue/`)
2. **Confidence threshold** — samples below this need review (default: 0.25)
3. **Annotation mode** — `label-studio` (GUI) or `cli` (command-line SAM)?

## Step 2: Analyze OOD Queue

```python
python - <<'EOF'
import json, glob
from collections import Counter
from pathlib import Path

ood_dir = Path("./ood_queue")
meta_files = sorted(ood_dir.glob("*.json"))
print(f"OOD queue: {len(meta_files)} samples pending review")

# Show class distribution of low-confidence detections
class_counts = Counter()
conf_sum = 0
for mf in meta_files:
    with open(mf) as f:
        data = json.load(f)
    for det in data.get("detections", []):
        class_counts[det["class"]] += 1
        conf_sum += det["confidence"]

if meta_files:
    print(f"Average confidence: {conf_sum / max(len(meta_files), 1):.3f}")
    print("Class distribution:")
    for cls, cnt in class_counts.most_common():
        print(f"  {cls}: {cnt}")
EOF
```

If queue is empty: "No OOD samples in queue yet. Run your edge inference service for a while to collect samples, then come back."

## Step 3: SAM-Assisted Annotation

### Option A: Label Studio (recommended for teams)

```bash
# Install if needed
pip install label-studio label-studio-ml

# Start Label Studio
label-studio start --port 8080 &

# Import OOD samples
echo "Open http://localhost:8080"
echo "1. Create project → Object Detection / Image Classification"
echo "2. Import from: $(pwd)/ood_queue/*.jpg"
echo "3. Enable SAM-assisted annotation in settings"
echo "4. Label each image, export as YOLO format when done"
```

### Option B: CLI with SAM (for solo use)

```bash
# Install SAM
pip install segment-anything
# Download SAM checkpoint (ViT-B for speed, ViT-H for accuracy)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O models/sam_vit_b.pth
```

```python
# sam_annotate.py — point-prompt segmentation for quick labeling
python - <<'EOF'
"""
Usage: python sam_annotate.py <image_path> <class_id>
Click on the object to segment it. Press 's' to save, 'n' for next, 'q' to quit.
"""
import sys
import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry

sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b.pth")
predictor = SamPredictor(sam)

def annotate_image(img_path, class_id):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    h, w = img.shape[:2]
    annotations = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            masks, _, _ = predictor.predict(
                point_coords=np.array([[x, y]]),
                point_labels=np.array([1]),
                multimask_output=False
            )
            mask = masks[0]
            # Convert mask to YOLO bbox
            ys, xs = np.where(mask)
            if len(xs) > 0:
                x1, y1 = xs.min() / w, ys.min() / h
                x2, y2 = xs.max() / w, ys.max() / h
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bw = x2 - x1
                bh = y2 - y1
                annotations.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                # Draw on image
                overlay = img.copy()
                overlay[mask] = [0, 255, 0]
                cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
                cv2.imshow("annotate", img)

    cv2.imshow("annotate", img)
    cv2.setMouseCallback("annotate", on_click)
    while True:
        key = cv2.waitKey(0)
        if key == ord('s'):
            label_path = img_path.replace('.jpg', '.txt').replace('.png', '.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(annotations))
            print(f"Saved {len(annotations)} annotations to {label_path}")
            break
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()

if len(sys.argv) >= 3:
    annotate_image(sys.argv[1], int(sys.argv[2]))
EOF
```

## Step 4: Merge New Annotations Into Dataset

```bash
python - <<'EOF'
import shutil, glob, os
from pathlib import Path
from datetime import datetime

ood_dir = Path("./ood_queue")
train_dir = Path("./data/train")
version_dir = Path("./data/versions") / datetime.now().strftime("%Y%m%d_%H%M%S")
version_dir.mkdir(parents=True, exist_ok=True)

# Snapshot current dataset version
for f in train_dir.glob("*"):
    shutil.copy(f, version_dir)

# Move labeled OOD samples to train
new_samples = 0
for img in ood_dir.glob("*.jpg"):
    label = img.with_suffix(".txt")
    if label.exists():
        shutil.move(str(img), str(train_dir / img.name))
        shutil.move(str(label), str(train_dir / label.name))
        new_samples += 1

# Write changelog
with open("data/versions/changelog.md", "a") as f:
    f.write(f"\n## {datetime.now().isoformat()}\n")
    f.write(f"Added {new_samples} new samples from active learning queue.\n")

print(f"Merged {new_samples} new samples into training set.")
print(f"Previous version backed up to: {version_dir}")
EOF
```

## Step 5: Trigger Incremental Finetune

```bash
# Finetune from best checkpoint (freeze backbone for small datasets)
yolo train \
  model=runs/<scenario>_latest/weights/best.pt \
  data=./data/data.yaml \
  epochs=30 \
  freeze=10 \
  imgsz=640 \
  batch=16 \
  device=<device> \
  project=./runs \
  name=<scenario>_finetune_$(date +%Y%m%d)
```

Note: `freeze=10` freezes the first 10 backbone layers. Use full finetune (`freeze=0`) if new data > 1000 images.

## Step 6: Compare Models

```python
python - <<'EOF'
import subprocess, json
from pathlib import Path

# Validate both models on val set
old_model = "runs/<scenario>_latest/weights/best.pt"
new_model = sorted(Path("runs").glob("*finetune*/weights/best.pt"))[-1]

def get_map(model_path):
    result = subprocess.run(
        ["yolo", "val", f"model={model_path}", "data=./data/data.yaml", "verbose=False"],
        capture_output=True, text=True
    )
    for line in result.stdout.split('\n'):
        if 'mAP50-95' in line or 'all' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == 'all' and i+4 < len(parts):
                    return float(parts[i+3]), float(parts[i+4])
    return 0, 0

old_map50, old_map = get_map(old_model)
new_map50, new_map = get_map(str(new_model))

print(f"Old model:  mAP50={old_map50:.3f}  mAP50-95={old_map:.3f}")
print(f"New model:  mAP50={new_map50:.3f}  mAP50-95={new_map:.3f}")
print(f"Delta:      mAP50={new_map50-old_map50:+.3f}  mAP50-95={new_map-old_map:+.3f}")

if new_map50 >= old_map50:
    print("IMPROVED — recommend deploying new model")
else:
    print("REGRESSED — do not deploy; investigate with yolo-rural:diagnose")
EOF
```

## Step 7: Auto-Deploy if Improved

If new model improves:
- Copy to `runs/<scenario>_latest/weights/best.pt`
- Invoke `yolo-rural:deploy` automatically, or ask user:
  > "New model improved by +X mAP50. Deploy to edge devices now? (yes/no)"

## Step 8: Report

```
Active learning cycle complete!
  New samples added:     <N>
  Old mAP50:             <value>
  New mAP50:             <value>
  Delta:                 <+/- value>
  Recommendation:        <deploy / investigate>

Dataset changelog: data/versions/changelog.md
```
````

**Step 2: Commit**

```bash
cd /Users/chenhao/MyOpenCode/yolo-skills
mkdir -p skills/yolo-rural/active-learning
cp ~/.claude/skills/yolo-rural/active-learning/SKILL.md skills/yolo-rural/active-learning/SKILL.md
git add -A
git commit -m "feat: add yolo-rural:active-learning skill"
```

---

## Task 7: Write Diagnose Skill

**Files:**
- Create: `~/.claude/skills/yolo-rural/diagnose/SKILL.md`

**Step 1: Write the file**

Create `~/.claude/skills/yolo-rural/diagnose/SKILL.md`:

````markdown
---
name: yolo-rural:diagnose
description: >
  Debug and analyze rural CV model performance. Use when user wants to understand
  why a model is failing, see metrics, view confusion matrix, find worst-performing
  classes/samples, or get improvement recommendations.
---

# yolo-rural:diagnose — Performance Diagnosis

## Step 1: Gather Inputs

Ask if not provided:
1. **Checkpoint path** — which model to diagnose? (default: `runs/<latest>/weights/best.pt`)
2. **Test data path** — where is the test set? (default: `./data/test/`)
3. **Scenario** — which scenario? (needed for threshold recommendations)

## Step 2: Run Full Validation

```bash
yolo val \
  model=<checkpoint> \
  data=./data/data.yaml \
  split=test \
  save_json=True \
  save_hybrid=True \
  plots=True \
  conf=0.001 \
  iou=0.6
```

Note: `conf=0.001` gives the full PR curve; filter by threshold in analysis.

## Step 3: Parse and Display Metrics

```python
python - <<'EOF'
import csv, os
from pathlib import Path

# Find most recent results.csv
results_files = sorted(Path("runs").glob("**/results.csv"), key=os.path.getmtime)
if not results_files:
    print("No results.csv found. Run training first.")
    exit()

with open(results_files[-1]) as f:
    rows = list(csv.DictReader(f))

last = rows[-1]
metrics = {
    "mAP50":       last.get("metrics/mAP50(B)", "N/A"),
    "mAP50-95":    last.get("metrics/mAP50-95(B)", "N/A"),
    "Precision":   last.get("metrics/precision(B)", "N/A"),
    "Recall":      last.get("metrics/recall(B)", "N/A"),
    "Box Loss":    last.get("val/box_loss", "N/A"),
    "Class Loss":  last.get("val/cls_loss", "N/A"),
}
print("Validation Metrics:")
for k, v in metrics.items():
    try:
        print(f"  {k:15s}: {float(v):.4f}")
    except:
        print(f"  {k:15s}: {v}")
EOF
```

## Step 4: Analyze Per-Class Performance

```python
python - <<'EOF'
import json
from pathlib import Path
from collections import defaultdict

# Parse COCO-format predictions if available
pred_file = sorted(Path("runs").glob("**/predictions.json"))
if pred_file:
    with open(pred_file[-1]) as f:
        preds = json.load(f)
    class_confs = defaultdict(list)
    for p in preds:
        class_confs[p.get("category_id", "unknown")].append(p["score"])
    print("\nPer-class confidence stats:")
    for cls_id, confs in sorted(class_confs.items()):
        avg = sum(confs) / len(confs)
        low = sum(1 for c in confs if c < 0.5)
        print(f"  Class {cls_id}: avg={avg:.3f}, low-conf={low}/{len(confs)}")
else:
    print("No predictions.json found. Check runs/ directory.")
EOF
```

## Step 5: Show Confusion Matrix Location

```bash
echo "Confusion matrix saved to:"
find runs/ -name "confusion_matrix*.png" | tail -3
echo ""
echo "PR curves saved to:"
find runs/ -name "*.png" | grep -E "PR|precision|recall" | tail -3
```

Tell user: "Open these PNG files to visually inspect class confusion and precision-recall tradeoffs."

## Step 6: Find Worst-Performing Samples

```python
python - <<'EOF'
import json, os
from pathlib import Path

pred_file = sorted(Path("runs").glob("**/predictions.json"))
if not pred_file:
    print("No predictions file. Run: yolo val ... save_json=True")
    exit()

with open(pred_file[-1]) as f:
    preds = json.load(f)

# Sort by confidence (lowest = worst performing)
worst = sorted(preds, key=lambda x: x.get("score", 1.0))[:10]
print("10 worst-performing predictions (lowest confidence):")
for p in worst:
    print(f"  img_id={p.get('image_id')} class={p.get('category_id')} conf={p.get('score', 0):.3f}")
EOF
```

## Step 7: Generate Recommendations

Based on the metrics, apply these rules and report findings:

```
IF mAP50 < 0.3:
  → "Critical: Model is barely learning. Check: (1) data quality — are labels correct?
     (2) class imbalance — do all classes have enough samples? (3) image quality."

IF mAP50 between 0.3 and 0.5:
  → "Low performance. Likely causes: insufficient data (need 500+ per class),
     high visual similarity between classes, or poor augmentation.
     Recommendation: run yolo-rural:active-learning to collect more data."

IF mAP50 between 0.5 and 0.7:
  → "Moderate. Improving to >0.7 requires more diverse data and possibly
     a larger model (yolov8m → yolov8l). Run active learning for 2-3 cycles."

IF mAP50 > 0.7:
  → "Good performance. Focus on recall for safety-critical classes
     (fire, fighting). Check per-class metrics for weakest class."

IF Recall < 0.5 for fire-safety or public-disorder:
  → "WARNING: Low recall on safety-critical scenario. Lower conf threshold to 0.15
     and accept more false positives to avoid missing real events."

IF Precision < 0.5:
  → "Too many false positives. Raise conf threshold or add more negative examples
     (background/non-target images) to training data."
```

## Step 8: Scenario-Specific Checks

**livestock-stray:**
- Check if detection works at night / in shadow (common failure mode)
- Verify zone polygon coordinates are correct in `deployment/edge_service.py`

**public-disorder:**
- Verify pose keypoints are being detected correctly
- Test with crowd scenes (false positives from normal gatherings)

**fire-safety:**
- Low recall is critical — verify `conf=0.15` or lower in production
- Test with reflection/glare (common false positive source)

**illegal-fishing:**
- Test at different times of day (dawn, dusk — common failure modes)
- Verify "require_both" logic (person AND rod must be detected)

**disease-recognition:**
- Check confusion between visually similar disease classes
- Verify image quality: lesion must be in focus, good lighting

## Step 9: Report to User

```
Diagnosis complete!
  mAP50:      <value>  [<rating: poor/moderate/good/excellent>]
  mAP50-95:   <value>
  Precision:  <value>
  Recall:     <value>

Key findings:
  <list of issues found>

Recommendations:
  1. <top recommendation>
  2. <second recommendation>

Plots saved to: runs/<run_name>/
  - confusion_matrix.png
  - PR_curve.png
  - results.png
```
````

**Step 2: Commit**

```bash
cd /Users/chenhao/MyOpenCode/yolo-skills
mkdir -p skills/yolo-rural/diagnose
cp ~/.claude/skills/yolo-rural/diagnose/SKILL.md skills/yolo-rural/diagnose/SKILL.md
git add -A
git commit -m "feat: add yolo-rural:diagnose skill"
```

---

## Task 8: Mirror Skills Into Repo

Keep `skills/` in the repo as a mirror of `~/.claude/skills/yolo-rural/` for version control.

**Step 1: Verify all files are in the repo**

```bash
ls -la /Users/chenhao/MyOpenCode/yolo-skills/skills/yolo-rural/
ls -la /Users/chenhao/MyOpenCode/yolo-skills/skills/yolo-rural/scaffold/
ls -la /Users/chenhao/MyOpenCode/yolo-skills/skills/yolo-rural/train/
ls -la /Users/chenhao/MyOpenCode/yolo-skills/skills/yolo-rural/deploy/
ls -la /Users/chenhao/MyOpenCode/yolo-skills/skills/yolo-rural/active-learning/
ls -la /Users/chenhao/MyOpenCode/yolo-skills/skills/yolo-rural/diagnose/
```

Expected: 6 SKILL.md files total.

**Step 2: Verify line counts (all under 400 lines)**

```bash
wc -l ~/.claude/skills/yolo-rural/*/SKILL.md ~/.claude/skills/yolo-rural/SKILL.md
```

**Step 3: Final commit**

```bash
cd /Users/chenhao/MyOpenCode/yolo-skills
git add -A
git commit -m "feat: complete yolo-rural skill suite (6 skills)"
```

---

## Task 9: Smoke Test

Verify the skills are discoverable and invocable in Claude Code.

**Step 1: Check skill registration**

```bash
ls ~/.claude/skills/yolo-rural/
ls ~/.claude/skills/yolo-rural/scaffold/
```

**Step 2: Mental smoke test — verify root skill routes correctly**

Open a new Claude Code session and type:
```
I want to set up a fire detection project for a rural area
```

Expected: Claude invokes `yolo-rural` (detects "fire detection" + "rural"), then routes to `yolo-rural:scaffold` with `scenario=fire-safety`.

**Step 3: Verify scaffold produces correct project structure**

```
Using yolo-rural:scaffold — scenario: fire-safety, target: jetson-orin
```

Expected output: project directory created with `data/`, `configs/`, `scripts/`, `deployment/`, `models/`.

**Step 4: Final status check**

```bash
echo "Skills installed:"
find ~/.claude/skills/yolo-rural -name "SKILL.md" | sort

echo ""
echo "Repo mirror:"
find /Users/chenhao/MyOpenCode/yolo-skills/skills -name "SKILL.md" | sort
```

Expected: 6 SKILL.md files in each location.

---

## Summary

| Task | Deliverable | Status |
|------|------------|--------|
| 1 | Directory structure | |
| 2 | `yolo-rural/SKILL.md` (root router) | |
| 3 | `scaffold/SKILL.md` | |
| 4 | `train/SKILL.md` | |
| 5 | `deploy/SKILL.md` | |
| 6 | `active-learning/SKILL.md` | |
| 7 | `diagnose/SKILL.md` | |
| 8 | Repo mirror | |
| 9 | Smoke test | |

---
name: yolo-rural:scaffold
description: >
  Scaffold a new rural CV project for a specific scenario. Use when user wants
  to create a new project, set up directory structure, download pretrained weights,
  or initialize a detection pipeline for livestock, fire, disorder, fishing, or disease.
  Triggered by: scaffold, create project, new project, set up, initialize, 牛马占道项目,
  明火项目, 病害项目.
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
python -c "import ultralytics; print('yolo')" 2>/dev/null && echo "framework=yolo" || true
python -c "import mmpretrain; print('mmpretrain')" 2>/dev/null || python -c "import mmcls; print('mmcls')" 2>/dev/null || true
```

If nothing installed:
- For real-time scenarios (livestock-stray, public-disorder, fire-safety, illegal-fishing): run `pip install ultralytics`
- For disease-recognition: run `pip install ultralytics` (YOLOv8-cls as fallback) or `pip install -U openmim && mim install mmpretrain`

## Step 3: Create Project Structure

```bash
PROJECT_DIR="<user-specified-or-default>"
mkdir -p "$PROJECT_DIR"/{data/{raw,train,val,test,versions},configs,scripts,models/pretrained,deployment,notebooks}
```

## Step 4: Generate data.yaml (YOLO format)

Create `<project-dir>/data/data.yaml` using the class definitions for the chosen scenario:

**livestock-stray:**
```yaml
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

Create `<project-dir>/configs/yolo_<scenario>.yaml`:

```yaml
# Model selection by scenario:
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
device: 0
workers: 8
project: ./runs
name: <scenario>_v1
save: true
val: true
plots: true
conf: 0.25
iou: 0.45
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
```

## Step 6: Generate Shell Scripts

Create `<project-dir>/scripts/train.sh`:
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

Create `<project-dir>/scripts/export.sh`:
```bash
#!/bin/bash
# Usage: ./scripts/export.sh <checkpoint.pt> <format: onnx|engine>
set -e
WEIGHTS=${1:-"runs/<scenario>_v1/weights/best.pt"}
FORMAT=${2:-"onnx"}
yolo export model=$WEIGHTS format=$FORMAT
echo "Exported to: ${WEIGHTS%.pt}.${FORMAT}"
```

Create `<project-dir>/scripts/infer.sh`:
```bash
#!/bin/bash
# Usage: ./scripts/infer.sh <source: image/video/rtsp_url>
set -e
WEIGHTS="runs/<scenario>_v1/weights/best.pt"
SOURCE=${1:-"0"}
yolo predict model=$WEIGHTS source=$SOURCE conf=0.25 save=True save_txt=True
```

Create `<project-dir>/scripts/annotate.sh`:
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
chmod +x <project-dir>/scripts/*.sh
```

## Step 7: Generate Deployment Files

Create `<project-dir>/deployment/Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN pip install ultralytics fastapi uvicorn python-multipart
COPY models/pretrained/ models/pretrained/
COPY deployment/edge_service.py .
EXPOSE 8000
CMD ["uvicorn", "edge_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `<project-dir>/deployment/edge_service.py`:
```python
"""Minimal inference API — OOD samples logged for active learning."""
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()
model = YOLO("models/pretrained/best.pt")
OOD_CONF_THRESHOLD = 0.25

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read()))
    results = model(img)
    detections, ood_samples = [], []
    for r in results:
        for box in r.boxes:
            det = {
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist(),
            }
            (ood_samples if float(box.conf) < OOD_CONF_THRESHOLD else detections).append(det)
    return {"detections": detections, "ood_queue": ood_samples}

@app.get("/health")
async def health():
    return {"status": "ok"}
```

Create `<project-dir>/deployment/alerts.py`:
```python
"""Scenario-specific alert rules — customize thresholds per deployment."""
ALERT_RULES = {
    "livestock-stray": {
        "classes": ["cattle", "horse"],
        "zone_crossing": True,
        "cooldown_seconds": 30,
    },
    "public-disorder": {
        "classes": ["fighting"],
        "min_persons": 2,
        "cooldown_seconds": 10,
    },
    "fire-safety": {
        "classes": ["flame", "smoke"],
        "zone_crossing": False,
        "cooldown_seconds": 5,
    },
    "illegal-fishing": {
        "classes": ["person", "fishing-rod"],
        "require_both": True,
        "time_window": "06:00-20:00",
        "cooldown_seconds": 60,
    },
    "disease-recognition": {
        "classes": ["skin-lesion", "respiratory", "digestive", "parasitic"],
        "batch_mode": True,
    },
}
```

## Step 8: Download Pretrained Weights

```python
cd "$PROJECT_DIR"
python - <<'EOF'
from ultralytics import YOLO
import os

scenario_models = {
    'livestock-stray': 'yolov8m.pt',
    'public-disorder': 'yolov8m-pose.pt',
    'fire-safety': 'yolov8n-seg.pt',
    'illegal-fishing': 'yolov8s.pt',
    'disease-recognition': 'yolov8s-cls.pt',
}
scenario = os.environ.get('SCENARIO', '<scenario>')
model_name = scenario_models.get(scenario, 'yolov8s.pt')
os.makedirs('models/pretrained', exist_ok=True)
os.chdir('models/pretrained')
YOLO(model_name)
print(f'Downloaded: {model_name}')
EOF
```

## Step 9: Generate README.md

Create `<project-dir>/README.md`:
```markdown
# <Scenario Name> — Rural CV Project

## Scenario
<Chinese name> | <English description>

## Minimum Dataset Size
- livestock-stray, fire-safety, illegal-fishing: 500+ images
- disease-recognition: 200+ images per class
- public-disorder: 1000+ sequences

## Quick Start
\`\`\`bash
# 1. Add images to data/raw/ and annotate
./scripts/annotate.sh        # Label Studio on :8080

# 2. Train
./scripts/train.sh

# 3. Export for edge
./scripts/export.sh runs/<scenario>_v1/weights/best.pt onnx

# 4. Run inference
./scripts/infer.sh /path/to/video.mp4
\`\`\`

## Next Steps
- Add more data:    yolo-rural:active-learning
- Deploy to edge:   yolo-rural:deploy
- Debug performance: yolo-rural:diagnose
```

## Step 10: Confirm to User

```
Project scaffolded at: <project_dir>/
  data/          — Add labeled images here (train/val/test splits, YOLO format)
  configs/       — Training config (yolo_<scenario>.yaml)
  scripts/       — train.sh  export.sh  infer.sh  annotate.sh
  models/        — Pretrained weights downloaded
  deployment/    — edge_service.py  Dockerfile  alerts.py

Next steps:
  1. Add labeled images to data/train/ and data/val/
     (or collect labels first: ./scripts/annotate.sh)
  2. Run: ./scripts/train.sh
  3. Deploy when ready: yolo-rural:deploy
```

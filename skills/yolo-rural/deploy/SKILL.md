---
name: yolo-rural:deploy
description: >
  Deploy a trained YOLO/OpenMMLab model to edge devices or cloud. Use when user
  wants to export, deploy, run on Jetson, Raspberry Pi, Docker, or package for
  production. Handles ONNX export, TensorRT optimization, systemd service, Docker.
  Triggered by: deploy, export, edge, jetson, raspberry pi, docker, production,
  部署, 导出, 边缘推理.
---

# yolo-rural:deploy — Edge & Cloud Deployment

## Step 1: Gather Inputs

Ask if not provided (one at a time):

1. **Checkpoint path** — which model? (default: find latest `runs/**/weights/best.pt`)
   ```bash
   find runs/ -name "best.pt" -newer /tmp/.deploy_sentinel 2>/dev/null | sort | tail -1
   # If no runs/ dir, ask user to provide checkpoint path
   ```

2. **Target platform** — where is this deploying?
   - `jetson-orin` — NVIDIA Jetson Orin (TensorRT + FP16, best edge performance)
   - `jetson-nano` — NVIDIA Jetson Nano (TensorRT + INT8, memory-constrained)
   - `raspberry-pi` — Raspberry Pi (ONNX Runtime, CPU-only)
   - `server` — On-premise GPU server (ONNX or native PyTorch)
   - `cloud` — Cloud API (Docker + FastAPI, horizontally scalable)

3. **Precision** — (default: fp16 for Jetson, fp32 for RPi/server, fp16 for cloud)
   - `fp32` — full precision, largest model
   - `fp16` — half precision, ~2x speedup on GPU
   - `int8` — 8-bit quantization, requires calibration data, best for Jetson Nano

4. **Scenario** — needed for alert rules and zone logic in the inference wrapper.

## Step 2: Find Checkpoint (if not provided)

```bash
CKPT=$(find runs/ -name "best.pt" | sort -t_ -k2 | tail -1)
echo "Using checkpoint: $CKPT"
```

## Step 3: Export Model

### Step 3a — ONNX (all platforms, do this first)

```bash
yolo export \
  model=<checkpoint> \
  format=onnx \
  imgsz=640 \
  half=<True if fp16/int8, False if fp32> \
  dynamic=False \
  simplify=True
# Output: <checkpoint_name>.onnx in same directory as checkpoint
```

### Step 3b — TensorRT engine (Jetson only — MUST run ON the Jetson device)

> IMPORTANT: TensorRT engines are device-specific. Do NOT build on the training server and copy to Jetson. Build directly on the target Jetson.

```bash
# Run this on the Jetson, not the training machine:
yolo export \
  model=<checkpoint> \
  format=engine \
  imgsz=640 \
  half=True \
  device=0
# Output: <checkpoint_name>.engine
```

### Step 3c — INT8 TensorRT (Jetson Nano, memory-constrained)

```bash
# Requires calibration data — provide representative images from data/val/
yolo export \
  model=<checkpoint> \
  format=engine \
  imgsz=640 \
  int8=True \
  data=./data/data.yaml \
  device=0
```

## Step 4: Generate Inference Service

Create or update `deployment/edge_service.py`:

```python
"""
Edge inference service with OOD flagging and scenario-specific alert rules.
Start: uvicorn deployment.edge_service:app --host 0.0.0.0 --port 8000
"""
import os, json, time, logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, BackgroundTasks
from ultralytics import YOLO
import io
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCENARIO = os.environ.get("SCENARIO", "<scenario>")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
OOD_THRESHOLD = float(os.environ.get("OOD_THRESHOLD", "0.25"))
OOD_LOG_DIR = Path(os.environ.get("OOD_LOG_DIR", "./ood_queue"))
OOD_LOG_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=f"yolo-rural:{SCENARIO}")
model = YOLO(MODEL_PATH)

def _log_ood(image: Image.Image, detections: list) -> None:
    ts = int(time.time())
    (OOD_LOG_DIR / f"{ts}.jpg").write_bytes(
        _img_to_bytes(image)
    )
    (OOD_LOG_DIR / f"{ts}.json").write_text(
        json.dumps({"timestamp": ts, "scenario": SCENARIO, "detections": detections})
    )

def _img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

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
                "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
            }
            (ood if float(box.conf) < OOD_THRESHOLD else detections).append(det)
    if ood:
        background_tasks.add_task(_log_ood, img, ood)
    return {
        "scenario": SCENARIO,
        "detections": detections,
        "ood_count": len(ood),
    }

@app.get("/health")
async def health():
    return {"status": "ok", "scenario": SCENARIO, "model": MODEL_PATH}
```

## Step 5: Platform Packaging

### Cloud / Server — Docker

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
ENV OOD_LOG_DIR=/app/ood_queue
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "edge_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t yolo-rural-<scenario>:latest -f deployment/Dockerfile .
docker run -d \
  -p 8000:8000 \
  -v "$(pwd)/ood_queue:/app/ood_queue" \
  --name rural-<scenario> \
  yolo-rural-<scenario>:latest
```

### Edge — systemd service (Jetson / RPi)

Create `deployment/rural-cv.service`:
```ini
[Unit]
Description=Rural CV Inference — <scenario>
After=network.target

[Service]
Type=simple
User=<deployment-user>
WorkingDirectory=<project-absolute-path>
Environment="SCENARIO=<scenario>"
Environment="MODEL_PATH=<project-absolute-path>/models/best.pt"
Environment="OOD_LOG_DIR=<project-absolute-path>/ood_queue"
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

### Push model update to edge (Ansible)

Create `deployment/deploy.yaml`:
```yaml
---
- hosts: edge_devices
  vars:
    project_path: /opt/rural-cv/<scenario>
  tasks:
    - name: Copy updated model weights
      copy:
        src: "{{ playbook_dir }}/../runs/best.pt"
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
# Health check
curl http://<device-ip>:8000/health

# Test prediction
curl -X POST http://<device-ip>:8000/predict \
  -F "file=@data/val/$(ls data/val/*.jpg 2>/dev/null | head -1)" \
  | python3 -m json.tool
```

Expected health response:
```json
{"status": "ok", "scenario": "<scenario>", "model": "models/best.pt"}
```

## Step 7: Report to User

```
Deployment complete!
  Format:    <onnx|engine|docker>
  Target:    <platform>
  Precision: <fp32|fp16|int8>
  Service:   http://<device-ip>:8000

Endpoints:
  POST /predict  — run inference (multipart image upload)
  GET  /health   — health check

OOD queue: <project>/ood_queue/
  Low-confidence detections are logged here automatically.
  Run yolo-rural:active-learning to review and retrain.
```

**Hardware-specific notes:**

- **Jetson Orin**: Use `format=engine half=True` — expect 50-200 FPS on YOLOv8s at 640px
- **Jetson Nano**: Use `format=engine int8=True` — expect 10-30 FPS; reduce imgsz to 416 if needed
- **Raspberry Pi 4**: Use ONNX Runtime; expect 2-5 FPS at 640px — reduce to 320px for real-time
- **On-premise server**: Use ONNX or native PyTorch; GPU servers (RTX 3080+) achieve 200+ FPS at 640px — no quantization needed
- **Cloud**: Docker + load balancer; scale replicas by traffic

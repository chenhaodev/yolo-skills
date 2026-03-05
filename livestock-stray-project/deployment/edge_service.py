"""
牛马占道 (Livestock Straying) — Cloud Inference API
Detects cattle and horses on roads/public areas.
OOD samples (low-confidence detections) are logged for active learning.

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

SCENARIO = "livestock-stray"
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
OOD_THRESHOLD = float(os.environ.get("OOD_THRESHOLD", "0.25"))
OOD_LOG_DIR = Path(os.environ.get("OOD_LOG_DIR", "./ood_queue"))
OOD_LOG_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="牛马占道检测 API", version="1.0.0")
model = YOLO(MODEL_PATH)

ALERT_CLASSES = {"cattle", "horse"}


def _log_ood(image: Image.Image, detections: list) -> None:
    """Save low-confidence sample for active learning review."""
    ts = int(time.time())
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    (OOD_LOG_DIR / f"{ts}.jpg").write_bytes(buf.getvalue())
    (OOD_LOG_DIR / f"{ts}.json").write_text(
        json.dumps({"timestamp": ts, "scenario": SCENARIO, "detections": detections})
    )
    logger.info("OOD sample logged: %s", ts)


@app.post("/predict")
async def predict(file: UploadFile, background_tasks: BackgroundTasks):
    """
    Run livestock detection on an uploaded image.

    Returns:
        detections: list of confirmed detections (conf >= threshold)
        ood_count: number of low-confidence detections logged for review
        alert: True if cattle or horse detected on road
    """
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))
    results = model(img, conf=OOD_THRESHOLD)

    detections, ood = [], []
    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            det = {
                "class": cls_name,
                "confidence": round(conf, 4),
                "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
            }
            if conf < OOD_THRESHOLD:
                ood.append(det)
            else:
                detections.append(det)

    if ood:
        background_tasks.add_task(_log_ood, img, ood)

    alert = any(d["class"] in ALERT_CLASSES for d in detections)

    logger.info("predict: %d detections, %d ood, alert=%s", len(detections), len(ood), alert)
    return {
        "scenario": SCENARIO,
        "detections": detections,
        "ood_count": len(ood),
        "alert": alert,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "scenario": SCENARIO, "model": MODEL_PATH}


@app.get("/ood/stats")
async def ood_stats():
    """Return count of pending OOD samples for active learning."""
    imgs = list(OOD_LOG_DIR.glob("*.jpg"))
    return {"pending_samples": len(imgs), "ood_dir": str(OOD_LOG_DIR)}

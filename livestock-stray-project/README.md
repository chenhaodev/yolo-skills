# 牛马占道检测 — Livestock Straying Detection

Detects cattle and horses straying into roads and public areas using YOLOv8m.

## Scenario
乡村养殖/牛马占道 | Livestock (cattle, horses) detected on roads or public zones

## Classes
- `cattle` (牛)
- `horse` (马)

## Minimum Dataset Size
- 500+ images minimum to start
- Diverse conditions: daytime/nighttime, rain, fog, different road types
- Both positive (animal on road) and negative (road without animals) samples

## Quick Start

```bash
# 1. Collect and annotate images
./scripts/annotate.sh        # Label Studio on http://localhost:8080

# 2. Train
./scripts/train.sh

# 3. Monitor training
tensorboard --logdir runs/

# 4. Export for deployment
./scripts/export.sh runs/livestock_stray_v1/weights/best.pt onnx

# 5. Run inference
./scripts/infer.sh /path/to/video.mp4

# 6. Deploy as cloud API
docker build -t livestock-stray:latest -f deployment/Dockerfile .
docker run -d -p 8000:8000 -v "$(pwd)/ood_queue:/app/ood_queue" livestock-stray:latest
```

## API Endpoints

```
POST /predict   — Upload image, returns detections + alert flag
GET  /health    — Health check
GET  /ood/stats — OOD queue size (samples ready for active learning)
```

## Zone Configuration

Edit `deployment/alerts.py` → `ZONE_POLYGONS["road_zone"]` to define the road
area polygon for your specific camera placement.

## Active Learning

Low-confidence detections are automatically logged to `./ood_queue/`.
When enough samples accumulate:

```bash
# Review and retrain
yolo-rural:active-learning
```

## Performance Targets
- mAP50 > 0.7 before deploying to production
- Recall > 0.85 (missing an animal on the road is a safety issue)
- Inference: ~50ms per frame on GPU, ~200ms on CPU

## Next Steps
- Collect data:   `./scripts/annotate.sh`
- Train:          `./scripts/train.sh`
- Deploy:         `yolo-rural:deploy`
- Improve:        `yolo-rural:active-learning`
- Debug:          `yolo-rural:diagnose`

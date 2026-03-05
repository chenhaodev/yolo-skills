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
python -c "import mmpretrain; print('MMPretrain:', mmpretrain.__version__)" 2>/dev/null || python -c "import mmcls; print('MMCls (legacy):', mmcls.__version__)" 2>/dev/null || echo "MMCls/MMPretrain: not installed"
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
> "Configured" = a `<scenario>-project/` directory exists in the current working directory.

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
- For disease recognition: recommend `pip install -U openmim && mim install mmpretrain` or `pip install ultralytics`

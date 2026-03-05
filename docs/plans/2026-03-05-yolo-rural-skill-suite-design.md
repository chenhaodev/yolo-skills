# YOLO Rural Skill Suite — Design Document

**Date:** 2026-03-05
**Status:** Approved, ready for implementation planning

---

## Overview

A modular Claude Code skill suite for building, finetuning, and deploying YOLO-based (and OpenMMLab-based) computer vision pipelines targeting rural/community daily life scenarios in China.

---

## Target Scenarios

| ID | Scenario | Chinese | Key Objects |
|----|----------|---------|-------------|
| 1 | Livestock straying into public areas | 乡村养殖/牛马占道 | Cattle, horses on roads/public zones |
| 2 | Public disorder detection | 公共区域/打架斗殴 | Fighting/brawling persons |
| 3 | Fire safety | 消防安全/明火识别 | Open flame, smoke |
| 4 | Illegal fishing | 河湖水库/垂钓检测 | Person + fishing rod at waterways |
| 5 | Livestock disease recognition | 乡村养殖/病害识别 | Animal skin/lesion patterns |

---

## Architecture Decision

**Chosen: Option B — Modular Skill Suite**

A family of focused skills under the `yolo-rural` namespace, each under 400 lines, composable and independently invocable.

Rejected alternatives:
- Option A (monolithic): becomes 800+ lines, hard to maintain
- Option C (scenario-driven agents): harder to reuse shared training/deploy components

---

## Skill Suite Structure

```
~/.claude/skills/yolo-rural/
├── SKILL.md                    # Root router: scenario overview, framework detection
├── scaffold/SKILL.md           # Project scaffolding for any scenario
├── train/SKILL.md              # Finetune/train (YOLO or OpenMMLab, auto-detected)
├── deploy/SKILL.md             # Edge export (ONNX/TensorRT) + cloud API packaging
├── active-learning/SKILL.md    # Collect → review → retrain loop
└── diagnose/SKILL.md           # Debug model performance, confusion matrix, error analysis
```

---

## Model Mapping Per Scenario

| Scenario | Task Type | Recommended Model | Edge Target | Notes |
|----------|-----------|-------------------|-------------|-------|
| livestock-stray | Detection + zone intrusion | YOLOv8m / YOLO11m | Jetson Nano/Orin | Polygon zone crossing logic |
| public-disorder | Pose estimation + action | YOLOv8-pose + classifier | Jetson Orin | Skeleton keypoint anomaly detection |
| fire-safety | Detection + segmentation | YOLOv8n-seg / YOLO11n | Any edge device | High recall priority, low latency |
| illegal-fishing | Person + object detection | YOLOv8s | IP camera | Time-of-day rules, nighttime IR support |
| disease-recognition | Fine-grained classification | MMClassification / YOLOv8-cls | Cloud or server | Accuracy > speed, batch inference OK |

### SAM Integration
- Used for **semi-automatic annotation** in the active learning loop (click → mask → label)
- Used for disease recognition: SAM segments lesion area, classifier identifies disease type
- NOT used for real-time inference on edge (too heavy)

### OOD Detection
- Each scenario includes confidence threshold + entropy-based OOD flag
- Low-confidence detections are automatically routed to the active learning queue

---

## Framework Detection Logic

Embedded in each skill, evaluated at runtime:
1. Check for `ultralytics` → use YOLO pipeline
2. Check for `mmdet`/`mmcls` → use OpenMMLab pipeline
3. Both present → prefer YOLO for real-time scenarios, OpenMMLab for disease/classification
4. Neither → scaffold installs the appropriate framework

---

## Deployment Target

**Hybrid:**
- **Edge**: Jetson Nano/Orin, Raspberry Pi, IP cameras with NPU — ONNX / TensorRT
- **Cloud/Server**: Training, management dashboard, active learning orchestration

---

## Skill Behaviors

### `yolo-rural` (root router)
- Lists 5 scenarios with configuration status
- Routes to appropriate sub-skill based on user intent
- Reports installed frameworks

### `yolo-rural:scaffold`
```
Inputs:  scenario name, project directory, framework preference
Actions:
  1. Detect installed frameworks
  2. Generate project structure (data/, configs/, scripts/, models/, deployment/)
  3. Download pretrained weights for chosen scenario
  4. Generate Label Studio config for annotation
```

### `yolo-rural:train`
```
Inputs:  data path, scenario, epochs, device, resume checkpoint
Actions:
  1. Validate dataset format (YOLO txt / COCO json / MM format)
  2. Auto-convert format if needed
  3. Generate/update config with data paths
  4. Run training with progress monitoring
  5. Evaluate on val set, generate metrics report
  6. Save best checkpoint with scenario metadata
```

### `yolo-rural:deploy`
```
Inputs:  checkpoint, target (jetson/rpi/cloud/docker), precision (fp16/int8)
Actions:
  1. Export to ONNX → TensorRT (Jetson) or ONNX Runtime (RPi/generic)
  2. Generate inference wrapper (zone logic, OOD flagging, alerts)
  3. Package as Docker container (cloud) or systemd service (edge)
  4. Generate deployment README with hardware-specific instructions
```

### `yolo-rural:active-learning`
```
Inputs:  inference logs dir, confidence threshold, annotation tool
Actions:
  1. Parse inference logs → extract low-confidence / OOD samples
  2. Launch SAM-assisted annotation (Label Studio or CLI)
  3. Merge new annotations into existing dataset
  4. Trigger incremental finetune via yolo-rural:train
  5. Compare new model vs previous on val set
  6. Auto-deploy if metrics improve
```

### `yolo-rural:diagnose`
```
Inputs:  checkpoint, test data, scenario
Actions:
  1. Run inference on test set
  2. Generate confusion matrix, PR curve, per-class metrics
  3. Show worst-performing samples visually
  4. Suggest: more data / augmentation / model size change
```

---

## Scaffolded Project Structure

When `yolo-rural:scaffold` is invoked, it generates:

```
<project-name>/
├── data/
│   ├── raw/                    # Original images
│   ├── train/ val/ test/       # YOLO-format splits
│   ├── versions/               # Dataset version snapshots
│   └── data.yaml               # YOLO dataset config
├── configs/
│   ├── yolo_<scenario>.yaml    # Ultralytics config
│   └── mm_<scenario>.py        # OpenMMLab config (if applicable)
├── scripts/
│   ├── train.sh                # One-command training
│   ├── export.sh               # Edge export (ONNX/TRT)
│   ├── infer.sh                # Inference with zone logic
│   └── annotate.sh             # Launch SAM + Label Studio
├── models/
│   └── pretrained/             # Downloaded base weights
├── deployment/
│   ├── Dockerfile              # Cloud/server container
│   ├── edge_service.py         # Edge inference service
│   ├── alerts.py               # Alert rules per scenario
│   └── deploy.yaml             # Ansible/SSH push config
├── notebooks/
│   └── explore.ipynb           # Data exploration + diagnosis
└── README.md                   # Scenario-specific guide
```

---

## Active Learning Data Pipeline

```
Edge Devices (camera → inference → alerts)
    ↓ low-confidence samples
Active Learning Queue (images + metadata + scores)
    ↓
SAM-assisted Annotation (click → mask → class label)
    ↓ new labeled data
Dataset Registry (versioned splits, YOLO format, data.yaml)
    ↓
Incremental Finetune (freeze backbone → train head, or full finetune if data >> 10k)
    ↓ if metrics improve
Auto-Deploy to Edge (export → push via SSH or container registry)
```

### Minimum Viable Dataset Sizes
- livestock-stray, fire-safety, illegal-fishing: 500+ images to start
- disease-recognition: 200+ images per class
- public-disorder: 1000+ sequences (action-based, needs temporal diversity)

### Data Conventions
- Primary format: YOLO txt
- Secondary export: COCO JSON (for OpenMMLab)
- Versions tracked in `data/versions/` with `changelog.md`

---

## Deliverables

6 SKILL.md files total:
1. `~/.claude/skills/yolo-rural/SKILL.md`
2. `~/.claude/skills/yolo-rural/scaffold/SKILL.md`
3. `~/.claude/skills/yolo-rural/train/SKILL.md`
4. `~/.claude/skills/yolo-rural/deploy/SKILL.md`
5. `~/.claude/skills/yolo-rural/active-learning/SKILL.md`
6. `~/.claude/skills/yolo-rural/diagnose/SKILL.md`

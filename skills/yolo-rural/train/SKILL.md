---
name: yolo-rural:train
description: >
  Train or finetune a YOLO/OpenMMLab model for a rural CV scenario. Use when user
  wants to train, finetune, retrain, or fine-tune a model. Works with Ultralytics
  YOLO and OpenMMLab frameworks. Validates data format, runs training, evaluates results.
  Triggered by: train, finetune, retrain, fine-tune, 训练, 微调.
---

# yolo-rural:train — Training & Finetuning

## Step 1: Gather Inputs

Ask if not provided (one at a time):

1. **Data path** — where is your dataset? (default: `./data`)
2. **Scenario** — which scenario? (livestock-stray / public-disorder / fire-safety / illegal-fishing / disease-recognition)
3. **Epochs** — how many epochs? (default: 100 for new training, 30 for finetune from checkpoint)
4. **Device** — GPU index or 'cpu'? (default: 0)
5. **Resume checkpoint** — resuming from a previous run? If yes, provide checkpoint path.

## Step 2: Detect Framework

```bash
python -c "import ultralytics; print('yolo:', ultralytics.__version__)" 2>/dev/null || echo "yolo: not installed"
python -c "import mmpretrain; print('mmpretrain: installed')" 2>/dev/null || python -c "import mmcls; print('mmcls: installed')" 2>/dev/null || echo "mmlab: not installed"
```

Framework selection:
- disease-recognition + mmpretrain/mmcls available → use OpenMMLab pipeline (Step 5B)
- all other cases → use YOLO pipeline (Step 5A)

## Step 3: Validate Dataset

```python
python - <<'EOF'
import os, glob

data_path = "<user-data-path>"
scenario = "<scenario>"
splits = ['train', 'val']
errors = []
for split in splits:
    img_dir = os.path.join(data_path, split)
    if not os.path.exists(img_dir):
        errors.append(f"ERROR: missing directory {img_dir}")
        continue
    imgs = glob.glob(f"{img_dir}/*.jpg") + glob.glob(f"{img_dir}/*.png") + glob.glob(f"{img_dir}/*.jpeg")
    txts = glob.glob(f"{img_dir}/*.txt")
    print(f"{split}: {len(imgs)} images, {len(txts)} label files")
    if len(imgs) == 0:
        errors.append(f"WARNING: no images in {img_dir}")
    if len(txts) == 0 and scenario != "disease-recognition":
        errors.append(f"WARNING: no label .txt files in {img_dir}")
for e in errors:
    print(e)
if not errors:
    print("Dataset validation passed.")
EOF
```

**If validation fails, stop and guide the user:**
- Missing directory: "Create `data/train/` and `data/val/`, add images and labels."
- No images: "Add .jpg/.png images to the split directory."
- No labels: "Label your images first — run `./scripts/annotate.sh` to launch Label Studio."
- Format mismatch (COCO JSON detected): proceed to Step 4 to auto-convert.

## Step 4: Auto-Convert COCO → YOLO Format (if needed)

Run only if dataset is in COCO JSON format (check for `annotations/*.json`):

```python
python - <<'EOF'
import json, os
from pathlib import Path

def coco_to_yolo(coco_json_path, output_dir):
    with open(coco_json_path) as f:
        coco = json.load(f)
    id_to_size = {img['id']: (img['width'], img['height']) for img in coco.get('images', [])}
    id_to_file = {img['id']: img['file_name'] for img in coco.get('images', [])}
    labels = {}
    skipped = 0
    for ann in coco.get('annotations', []):
        img_id = ann.get('image_id')
        if img_id not in id_to_size:
            skipped += 1
            continue
        bbox = ann.get('bbox')
        if not bbox or len(bbox) != 4:
            skipped += 1
            continue
        w, h = id_to_size[img_id]
        if w == 0 or h == 0:
            skipped += 1
            continue
        x, y, bw, bh = bbox
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        cls = ann.get('category_id', 1) - 1
        stem = Path(id_to_file[img_id]).stem
        labels.setdefault(stem, []).append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    os.makedirs(output_dir, exist_ok=True)
    for stem, lines in labels.items():
        with open(f"{output_dir}/{stem}.txt", 'w') as f:
            f.write('\n'.join(lines))
    print(f"Converted {len(labels)} annotation files → {output_dir} (skipped {skipped} invalid annotations)")

# Update paths as needed:
# coco_to_yolo("data/annotations/instances_train2017.json", "data/train")
# coco_to_yolo("data/annotations/instances_val2017.json", "data/val")
EOF
```

## Step 5A: Run YOLO Training

```bash
# Determine model for scenario:
# livestock-stray:     yolov8m.pt
# public-disorder:     yolov8m-pose.pt
# fire-safety:         yolov8n-seg.pt
# illegal-fishing:     yolov8s.pt
# disease-recognition: yolov8s-cls.pt  (fallback if mmpretrain not available)

yolo train \
  model=<model-for-scenario> \
  data=<data-path>/data.yaml \
  epochs=<epochs> \
  imgsz=640 \
  batch=16 \
  device=<device> \
  project=./runs \
  name=<scenario>_v$(date +%Y%m%d) \
  save=True \
  plots=True \
  conf=0.25 \
  iou=0.45
```

For finetune from checkpoint (resume=True if `--resume` flag, else specify weights):
```bash
yolo train \
  model=<checkpoint-path> \
  data=<data-path>/data.yaml \
  epochs=<epochs> \
  freeze=10 \
  device=<device> \
  project=./runs \
  name=<scenario>_finetune_$(date +%Y%m%d)
```

Note: `freeze=10` freezes the first 10 backbone layers — appropriate for small new datasets (<1000 images). Use `freeze=0` for full finetune with large new datasets (>1000 images).

## Step 5B: Run OpenMMLab Training (disease-recognition)

Generate `configs/mm_disease_recognition.py` if it doesn't exist:

```python
python - <<'EOF'
config = """
_base_ = ['mmpretrain::resnet/resnet50_8xb32_in1k.py']

model = dict(
    head=dict(num_classes=5)  # adjust to actual class count
)

data_root = './data/'
train_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale=224),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs'),
        ]
    )
)
val_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs'),
        ]
    )
)
val_evaluator = dict(type='Accuracy', topk=(1,))
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=5)
"""
with open('configs/mm_disease_recognition.py', 'w') as f:
    f.write(config)
print("Generated configs/mm_disease_recognition.py")
EOF
```

Run training:
```bash
python -m mmpretrain.tools.train configs/mm_disease_recognition.py \
  --work-dir ./runs/disease_$(date +%Y%m%d)
```

## Step 6: Post-Training Evaluation

After training completes:

```python
python - <<'EOF'
import csv, os
from pathlib import Path

results_files = sorted(Path('runs').glob('**/results.csv'), key=os.path.getmtime)
if not results_files:
    print("No results.csv found yet — training may still be running.")
else:
    with open(results_files[-1]) as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    keys = {
        'mAP50':      'metrics/mAP50(B)',
        'mAP50-95':   'metrics/mAP50-95(B)',
        'Precision':  'metrics/precision(B)',
        'Recall':     'metrics/recall(B)',
    }
    print("Validation metrics (best checkpoint):")
    for label, key in keys.items():
        val = last.get(key, 'N/A')
        try:
            print(f"  {label:12s}: {float(val):.4f}")
        except (ValueError, TypeError):
            print(f"  {label:12s}: {val}")
    print(f"\nBest checkpoint: {results_files[-1].parent}/weights/best.pt")
EOF
```

## Step 7: Report to User

```
Training complete!
  Checkpoint:  runs/<scenario>_<date>/weights/best.pt
  mAP50:       <value>
  mAP50-95:    <value>
  Precision:   <value>
  Recall:      <value>

Next steps:
  - Deploy to edge:         yolo-rural:deploy
  - Diagnose failures:      yolo-rural:diagnose
  - Add more data:          yolo-rural:active-learning
```

If mAP50 < 0.5, proactively add:
> "Performance is below 0.5 mAP50. Consider: (1) adding more diverse training images via `yolo-rural:active-learning`, (2) running `yolo-rural:diagnose` to identify failure modes, or (3) increasing model size (e.g. yolov8s → yolov8m)."

#!/bin/bash
# Run inference on image / video / RTSP stream
# Usage: ./scripts/infer.sh <source>
#   source: path to image, video file, or rtsp://... URL
set -e

WEIGHTS="runs/livestock_stray_v1/weights/best.pt"
SOURCE=${1:-"0"}   # 0 = webcam/first camera

echo "Running inference: $SOURCE"
yolo predict \
    model=$WEIGHTS \
    source=$SOURCE \
    conf=0.25 \
    save=True \
    save_txt=True \
    line_width=2

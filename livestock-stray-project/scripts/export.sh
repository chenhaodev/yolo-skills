#!/bin/bash
# Export model to ONNX or TensorRT
# Usage: ./scripts/export.sh [checkpoint.pt] [format: onnx|engine]
set -e

WEIGHTS=${1:-"runs/livestock_stray_v1/weights/best.pt"}
FORMAT=${2:-"onnx"}

echo "Exporting: $WEIGHTS → $FORMAT"
yolo export model=$WEIGHTS format=$FORMAT imgsz=640 simplify=True
echo "Done: ${WEIGHTS%.pt}.${FORMAT}"

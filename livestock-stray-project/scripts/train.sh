#!/bin/bash
# Train livestock-stray detection model
# Usage: ./scripts/train.sh [--resume]
set -e

CONFIG="configs/yolo_livestock_stray.yaml"

if [ "$1" = "--resume" ]; then
    echo "Resuming training from last checkpoint..."
    yolo train cfg=$CONFIG resume=True
else
    echo "Starting fresh training..."
    yolo train cfg=$CONFIG
fi

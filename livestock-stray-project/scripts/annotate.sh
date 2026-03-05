#!/bin/bash
# Launch Label Studio for data annotation
# Requires: pip install label-studio
set -e

echo "Starting Label Studio on http://localhost:8080"
echo "Import images from: $(pwd)/data/raw/"
echo ""
echo "Annotation instructions:"
echo "  1. Create project → Object Detection (Bounding Box)"
echo "  2. Classes: cattle, horse"
echo "  3. Import from $(pwd)/data/raw/"
echo "  4. Label images, then Export → YOLO format"
echo "  5. Move exported images+labels to data/train/ and data/val/"
echo ""
label-studio start --port 8080

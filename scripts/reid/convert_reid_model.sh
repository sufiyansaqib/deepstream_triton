#!/bin/bash

# Convert NVIDIA ReID model to TensorRT engine
# This script builds the ReID model inside the DeepStream container for compatibility

set -e

echo "Converting NVIDIA ReID model to TensorRT engine..."

# Check if model exists
if [ ! -f "models/reid/reid_model/1/model.etlt" ]; then
    echo "❌ ReID model not found at models/reid/reid_model/1/model.etlt"
    exit 1
fi

echo "Building TensorRT engine inside DeepStream container..."

# Build the TensorRT engine using DeepStream container
docker run --rm --gpus all \
  -v $(pwd)/models:/workspace/models \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  bash -c "
    cd /workspace && 
    echo 'Converting ReID model to TensorRT engine...' &&
    /usr/src/tensorrt/bin/trtexec \
      --onnx=/workspace/models/reid/reid_model/1/model.etlt \
      --saveEngine=/workspace/models/reid/reid_model/1/model.plan \
      --fp16 \
      --workspace=1024 \
      --minShapes=input:1x3x256x128 \
      --optShapes=input:50x3x256x128 \
      --maxShapes=input:100x3x256x128 \
      --verbose
  "

# Verify the engine was created
if [ -f "models/reid/reid_model/1/model.plan" ]; then
    echo "✅ ReID TensorRT engine created successfully!"
    echo "   Engine size: $(ls -lh models/reid/reid_model/1/model.plan | awk '{print $5}')"
    echo "   Location: $(pwd)/models/reid/reid_model/1/model.plan"
else
    echo "❌ Failed to create TensorRT engine"
    exit 1
fi

echo "🎉 ReID model conversion complete!"
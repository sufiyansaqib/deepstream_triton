#!/bin/bash

# Build custom YOLOv7 parsing library for DeepStream
# This script builds the custom parsing function inside a DeepStream development container

set -e

echo "Building custom YOLOv7 parsing library..."

# Get CUDA version from system
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9.]*\),.*/\1/' | head -1)
if [ -z "$CUDA_VER" ]; then
    CUDA_VER="12.1"  # Default fallback
    echo "Warning: Could not detect CUDA version, using default $CUDA_VER"
else
    echo "Detected CUDA version: $CUDA_VER"
fi

# Build using DeepStream development container
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace/nvdsinfer_custom_impl_yolov7 \
    nvcr.io/nvidia/deepstream:7.1-gc-triton-devel \
    bash -c "
        echo 'Building custom YOLOv7 parsing library...' && \
        make clean && \
        CUDA_VER=$CUDA_VER make -j\$(nproc) && \
        echo 'Build completed successfully!' && \
        ls -la *.so
    "

echo "Custom library built successfully!"
echo "Location: nvdsinfer_custom_impl_yolov7/libnvdsinfer_custom_impl_yolov7.so"
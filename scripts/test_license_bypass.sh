#!/bin/bash

set -e

echo "🚀 Testing DeepStream License Bypass Solutions..."

# Create output directory with timestamp
OUTPUT_DIR="output/license_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"

# Method 1: Set environment variable to accept license
echo "🔧 Method 1: Testing with license environment variable..."
docker run --rm --gpus all \
  -v ./videos:/workspace/videos \
  -v ./configs:/workspace/configs \
  -v ./"$OUTPUT_DIR":/workspace/output \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ACCEPT_EULA=Y \
  --name deepstream-license-test1 \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  timeout 30 bash -c "
    cd /workspace && 
    echo 'Testing with ACCEPT_EULA=Y...' &&
    deepstream-app --version 2>&1 | head -10
  " || echo "Method 1 failed"

echo "🔧 Method 2: Testing with different approach..."
docker run --rm --gpus all \
  -v ./videos:/workspace/videos \
  -v ./configs:/workspace/configs \
  -v ./"$OUTPUT_DIR":/workspace/output \
  --name deepstream-license-test2 \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  timeout 30 bash -c "
    echo 'Testing different approach...' &&
    echo 'yes' | deepstream-app --version 2>&1 | head -10
  " || echo "Method 2 failed"

echo "🔧 Method 3: Testing with stdin redirect..."
echo "yes" | docker run --rm --gpus all -i \
  -v ./videos:/workspace/videos \
  -v ./configs:/workspace/configs \
  -v ./"$OUTPUT_DIR":/workspace/output \
  --name deepstream-license-test3 \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  timeout 30 bash -c "
    echo 'Testing with stdin redirect...' &&
    deepstream-app --version 2>&1 | head -10
  " || echo "Method 3 failed"

echo "🏁 License bypass tests completed! Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"

echo "✅ License bypass test finished!"
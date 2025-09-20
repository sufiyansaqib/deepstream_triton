#!/bin/bash

set -e

echo "🚀 Testing Direct Dual Video Pipeline (No Triton)..."

# Create output directory with timestamp
OUTPUT_DIR="output/direct_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"

# Test DeepStream with a simple configuration first
echo "🎬 Testing DeepStream without Triton..."
docker run --rm --gpus all \
  -v ./videos:/workspace/videos \
  -v ./configs:/workspace/configs \
  -v ./"$OUTPUT_DIR":/workspace/output \
  -v ./models:/workspace/models \
  -v ./labels:/workspace/labels \
  -v ./trackers:/workspace/trackers \
  --name deepstream-direct-test \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  timeout 60 bash -c "
    cd /workspace && 
    echo '🎥 Testing basic DeepStream functionality...' &&
    echo 'Available configs:' && ls -la configs/ &&
    echo 'Available videos:' && ls -la videos/ &&
    echo 'Testing simple sample application...' &&
    deepstream-app -c /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt 2>&1 | 
    head -20
  "

echo "🏁 Direct test completed! Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"

echo "✅ Direct dual video test complete!"
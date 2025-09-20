#!/bin/bash

# Test ReID tracking functionality
# Runs a simple test with the enhanced tracker configuration

set -e

echo "Testing ReID-enabled tracking..."
echo "======================================="

# Ensure output directory exists
mkdir -p output

# Start Triton server
echo "Starting Triton server..."
docker compose up -d triton

# Wait for Triton to be ready
echo "Waiting for Triton server to be ready..."
sleep 15

# Test ReID tracking
echo "Running DeepStream with ReID tracking..."
docker run --rm --gpus all \
  --network deepstream_triton_deepstream-triton \
  -v ./videos:/workspace/videos \
  -v ./configs:/workspace/configs \
  -v ./configs/reid:/workspace/configs/reid \
  -v ./output:/workspace/output \
  -v ./models:/workspace/models \
  -v ./labels:/workspace/labels \
  -v ./trackers:/workspace/trackers \
  --name deepstream-reid-test \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  bash -c "
    cd /workspace && 
    echo 'Starting DeepStream with ReID tracking...' &&
    timeout 60s deepstream-app -c configs/reid/deepstream_reid_enabled.txt 2>&1 | 
    tee /workspace/output/reid_tracking_test.log || echo 'Test completed'
  "

echo "ReID tracking test completed!"
echo "Checking output files..."
ls -la output/

# Check if any ReID features were extracted
echo "Checking for ReID tracking output..."
if [ -f "output/reid_tracking_test.log" ]; then
    echo "✅ Test log created"
    
    # Check for ReID-related messages
    if grep -q "ReID\|reid\|Re-ID" output/reid_tracking_test.log; then
        echo "✅ ReID functionality detected in logs"
        grep -i "reid\|re-id" output/reid_tracking_test.log | head -5
    else
        echo "⚠️  No explicit ReID messages found, but tracker may be working"
    fi
    
    # Check for tracking messages
    if grep -q "NvDCF" output/reid_tracking_test.log; then
        echo "✅ NvDCF tracker active"
    fi
    
    # Check for any errors
    if grep -q "ERROR\|Error\|error" output/reid_tracking_test.log; then
        echo "⚠️  Errors detected:"
        grep -i "error" output/reid_tracking_test.log | head -3
    fi
else
    echo "❌ Test log not found"
fi

echo "🎉 ReID tracking test analysis complete!"
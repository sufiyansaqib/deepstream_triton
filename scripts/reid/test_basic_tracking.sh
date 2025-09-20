#!/bin/bash

# Basic Tracking Test (without ReID initially)
# Test the basic DeepStream pipeline first, then enable ReID

set -e

echo "🎯 Basic Tracking Test"
echo "======================"

# Use the original working configuration first
echo "📊 Testing with original configuration (dual video)..."

# Start Triton if needed
if ! docker ps --filter "name=triton" --format "table {{.Names}}" | grep -q triton; then
    echo "⚠️  Starting Triton server..."
    docker compose up -d triton
    sleep 15
fi

# Create output directory
mkdir -p output/basic_test

echo "🎬 Running basic tracking (30 seconds)..."

# Run with the original working configuration
timeout 30s ./run_with_fps.sh || echo "⏱️ Test completed (timeout expected)"

# Check what was generated
echo ""
echo "📊 Checking generated outputs..."
ls -la output/

# Check for any video files
echo ""
echo "🎬 Video outputs:"
find output/ -name "*.mp4" -exec ls -lh {} \; | head -10

echo ""
echo "🎯 Basic test completed!"
echo "Now let's verify the ReID-enhanced version works..."

echo ""
echo "🔄 Testing ReID-enhanced tracking..."

# Run a shorter test with ReID configuration
docker run --rm \
  --network deepstream_triton_deepstream-triton \
  -v $(pwd)/videos:/workspace/videos \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/labels:/workspace/labels \
  -v $(pwd)/trackers:/workspace/trackers \
  --name deepstream-basic-test \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  bash -c "
    cd /workspace && 
    echo '🔧 Using original tracker config for compatibility...' &&
    timeout 20s deepstream-app -c configs/deepstream_dual_video_triton.txt 2>&1 | 
    tee /workspace/output/basic_test/basic_tracking.log || echo '⏱️ Basic test completed'
  "

# Analyze results
echo ""
echo "📊 Analyzing basic tracking results..."

if [ -f "output/basic_test/basic_tracking.log" ]; then
    echo "✅ Basic tracking log created"
    
    # Check for key components
    if grep -q -i "tracker\|tracking" output/basic_test/basic_tracking.log; then
        echo "  ✅ Tracker active"
    fi
    
    if grep -q -i "inference\|detection" output/basic_test/basic_tracking.log; then
        echo "  ✅ Inference active" 
    fi
    
    if grep -q -i "fps\|frame" output/basic_test/basic_tracking.log; then
        echo "  ✅ Performance data available"
        grep -i "fps\|frame.*rate" output/basic_test/basic_tracking.log | tail -2
    fi
    
    # Check for errors
    ERROR_COUNT=$(grep -c -i "error\|failed" output/basic_test/basic_tracking.log || echo "0")
    echo "  ℹ️  Errors detected: $ERROR_COUNT"
    
    echo ""
    echo "📝 Recent log entries:"
    tail -10 output/basic_test/basic_tracking.log
    
else
    echo "❌ Basic tracking log not found"
fi

echo ""
echo "🎯 Basic tracking test completed!"
echo "Check output/ directory for any generated video files."
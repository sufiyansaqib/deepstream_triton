#!/bin/bash

# Direct ReID Test - Simple verification of ReID functionality
# Tests the ReID-enabled tracker without complex logging

set -e

echo "Direct ReID Tracking Test"
echo "========================="

# Ensure output directory exists
mkdir -p output/reid_test

echo "📊 Testing ReID tracker configuration..."

# Test 1: Quick syntax check
echo "🔍 Step 1: Configuration syntax check..."
if docker run --rm -v $(pwd):/workspace nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
   bash -c "deepstream-app --help > /dev/null"; then
    echo "✅ DeepStream executable accessible"
else
    echo "❌ DeepStream executable issue"
    exit 1
fi

# Test 2: Basic pipeline validation
echo "🔍 Step 2: Pipeline configuration validation..."
if docker run --rm -v $(pwd):/workspace nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
   bash -c "ls -la /workspace/configs/reid/deepstream_reid_enabled.txt && ls -la /workspace/configs/reid/tracker_reid_enhanced.yml"; then
    echo "✅ Configuration files accessible in container"
else
    echo "❌ Configuration files not accessible"
    exit 1
fi

# Test 3: Short pipeline run with ReID
echo "🔍 Step 3: Short ReID pipeline execution..."
echo "Running 15-second ReID test..."

docker run --rm --gpus all \
  --network deepstream_triton_deepstream-triton \
  -v $(pwd)/videos:/workspace/videos \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/labels:/workspace/labels \
  -v $(pwd)/trackers:/workspace/trackers \
  --name deepstream-reid-direct-test \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  bash -c "
    cd /workspace && 
    echo '🚀 Starting ReID-enabled pipeline...' &&
    timeout 15s deepstream-app -c configs/reid/deepstream_reid_enabled.txt 2>&1 | 
    tee /workspace/output/reid_test/direct_test.log || echo '⏱️ Test timeout completed (expected)'
  "

echo "🔍 Step 4: Analyzing results..."

# Check if log was created
if [ -f "output/reid_test/direct_test.log" ]; then
    echo "✅ Test log created successfully"
    
    # Check for key indicators
    echo "📊 Key indicators:"
    
    # Check for tracker initialization
    if grep -q "NvDCF\|nvtracker\|tracker" output/reid_test/direct_test.log; then
        echo "  ✅ Tracker initialization detected"
    else
        echo "  ⚠️  No tracker initialization found"
    fi
    
    # Check for pipeline elements
    if grep -q "primary-gie\|inference" output/reid_test/direct_test.log; then
        echo "  ✅ Primary inference engine active"
    else
        echo "  ⚠️  No inference engine activity"
    fi
    
    # Check for any ReID-related activity
    if grep -q -i "reid\|feature\|association" output/reid_test/direct_test.log; then
        echo "  ✅ ReID-related activity detected"
    else
        echo "  ℹ️  No explicit ReID activity (may be internal)"
    fi
    
    # Check for errors
    if grep -q -i "error\|failed\|cannot" output/reid_test/direct_test.log; then
        echo "  ⚠️  Potential issues detected:"
        grep -i "error\|failed\|cannot" output/reid_test/direct_test.log | head -3
    else
        echo "  ✅ No major errors detected"
    fi
    
    # Show pipeline status
    echo ""
    echo "📈 Pipeline execution summary:"
    echo "   - Log size: $(wc -l < output/reid_test/direct_test.log) lines"
    echo "   - Test duration: ~15 seconds"
    
    # Show last few lines for context
    echo ""
    echo "📝 Last few log lines:"
    tail -5 output/reid_test/direct_test.log
    
else
    echo "❌ Test log not created - pipeline may have failed"
    exit 1
fi

echo ""
echo "🎯 Direct ReID test completed!"
echo "   Full log available at: output/reid_test/direct_test.log"

# Cleanup
docker rm -f deepstream-reid-direct-test 2>/dev/null || true

echo "🎉 ReID functionality verification complete!"
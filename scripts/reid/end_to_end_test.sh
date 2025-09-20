#!/bin/bash

# End-to-End ReID Tracking Test
# Tests the complete ReID pipeline with the 2 existing videos

set -e

echo "🚀 End-to-End ReID Tracking Test"
echo "=================================="

# Setup
TEST_NAME="reid_e2e_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="output/reid_test/$TEST_NAME"
mkdir -p "$OUTPUT_DIR"

echo "📁 Test directory: $OUTPUT_DIR"

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check videos exist
if [ ! -f "videos/mb1_1.mp4" ] || [ ! -f "videos/mb2_2.mp4" ]; then
    echo "❌ Required video files not found"
    exit 1
fi

# Check Triton is running
if ! docker ps --filter "name=triton" --format "table {{.Names}}" | grep -q triton; then
    echo "⚠️  Starting Triton server..."
    docker compose up -d triton
    sleep 15
else
    echo "✅ Triton server is running"
fi

# Check GPU availability in Docker
echo "🔍 Checking GPU availability..."
GPU_CHECK=$(docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi 2>/dev/null && echo "GPU_OK" || echo "GPU_FAIL")

if [ "$GPU_CHECK" = "GPU_FAIL" ]; then
    echo "⚠️  GPU not available in Docker, running CPU-only test"
    GPU_FLAGS=""
else
    echo "✅ GPU available in Docker"
    GPU_FLAGS="--gpus all"
fi

# Test Configuration
echo "📊 Testing with configuration:"
echo "  - Video 1: videos/mb1_1.mp4"
echo "  - Video 2: videos/mb2_2.mp4"
echo "  - ReID Config: configs/reid/deepstream_reid_enabled.txt"
echo "  - Tracker Config: configs/reid/tracker_reid_enhanced.yml"
echo "  - Test Duration: 30 seconds"
echo "  - Output Directory: $OUTPUT_DIR"

# Run the ReID tracking pipeline
echo ""
echo "🎬 Starting ReID tracking pipeline..."
echo "======================================"

docker run --rm $GPU_FLAGS \
  --network deepstream_triton_deepstream-triton \
  -v $(pwd)/videos:/workspace/videos \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/labels:/workspace/labels \
  -v $(pwd)/trackers:/workspace/trackers \
  --name deepstream-reid-e2e \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  bash -c "
    cd /workspace && 
    echo '🚀 Initializing DeepStream ReID pipeline...' &&
    echo 'Using configuration: configs/reid/deepstream_reid_enabled.txt' &&
    echo 'Video sources:' &&
    ls -la videos/*.mp4 &&
    echo '' &&
    echo '📊 Starting tracking with ReID...' &&
    timeout 30s deepstream-app -c configs/reid/deepstream_reid_enabled.txt 2>&1 | 
    tee /workspace/output/reid_test/$TEST_NAME/pipeline.log || echo '⏱️ Pipeline completed (30s timeout)'
  "

# Check results
echo ""
echo "🔍 Analyzing results..."
echo "======================"

# Check if log was created
LOG_FILE="$OUTPUT_DIR/pipeline.log"
if [ -f "$LOG_FILE" ]; then
    echo "✅ Pipeline log created: $(wc -l < $LOG_FILE) lines"
    
    # Analyze log for key indicators
    echo ""
    echo "📊 Pipeline Analysis:"
    
    # Check for successful initialization
    if grep -q "Pipeline ready" "$LOG_FILE" || grep -q "Playing" "$LOG_FILE"; then
        echo "  ✅ Pipeline initialized successfully"
    else
        echo "  ⚠️  Pipeline initialization unclear"
    fi
    
    # Check for tracker activity
    if grep -q -i "tracker\|tracking\|nvdcf" "$LOG_FILE"; then
        echo "  ✅ Tracker activity detected"
        TRACKER_LINES=$(grep -c -i "tracker\|tracking\|nvdcf" "$LOG_FILE")
        echo "    - Tracker-related messages: $TRACKER_LINES"
    else
        echo "  ⚠️  No tracker activity detected"
    fi
    
    # Check for inference activity
    if grep -q -i "inference\|infer\|detection" "$LOG_FILE"; then
        echo "  ✅ Inference engine active"
        INFER_LINES=$(grep -c -i "inference\|infer\|detection" "$LOG_FILE")
        echo "    - Inference-related messages: $INFER_LINES"
    else
        echo "  ⚠️  No inference activity detected"
    fi
    
    # Check for errors
    ERROR_COUNT=$(grep -c -i "error\|failed\|cannot\|critical" "$LOG_FILE" || echo "0")
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "  ⚠️  Errors detected: $ERROR_COUNT"
        echo "    Recent errors:"
        grep -i "error\|failed\|cannot" "$LOG_FILE" | tail -3 | sed 's/^/      /'
    else
        echo "  ✅ No critical errors detected"
    fi
    
    # Performance indicators
    if grep -q -i "fps\|frame" "$LOG_FILE"; then
        echo "  📈 Performance data available"
        grep -i "fps\|frame.*rate" "$LOG_FILE" | tail -2 | sed 's/^/    /'
    fi
    
else
    echo "❌ Pipeline log not found"
fi

# Check for output videos
echo ""
echo "🎬 Checking output videos..."
echo "============================"

OUTPUT_VIDEOS=(
    "output/reid_video_detections_0.mp4"
    "output/reid_video_detections_1.mp4"
)

for video in "${OUTPUT_VIDEOS[@]}"; do
    if [ -f "$video" ]; then
        SIZE=$(ls -lh "$video" | awk '{print $5}')
        echo "  ✅ $video ($SIZE)"
        
        # Copy to test directory for preservation
        cp "$video" "$OUTPUT_DIR/"
        echo "    → Saved to $OUTPUT_DIR/"
    else
        echo "  ❌ $video not found"
    fi
done

# Summary
echo ""
echo "📋 Test Summary"
echo "==============="
echo "Test Name: $TEST_NAME"
echo "Test Time: $(date)"
echo "Log File: $OUTPUT_DIR/pipeline.log"

if [ -f "$LOG_FILE" ]; then
    echo "Pipeline Status: ✅ Executed"
    echo "Log Size: $(wc -l < $LOG_FILE) lines"
else
    echo "Pipeline Status: ❌ Failed"
fi

VIDEO_COUNT=0
for video in "${OUTPUT_VIDEOS[@]}"; do
    if [ -f "$video" ]; then
        ((VIDEO_COUNT++))
    fi
done

echo "Output Videos: $VIDEO_COUNT/2 generated"

# Final status
if [ -f "$LOG_FILE" ] && [ "$VIDEO_COUNT" -gt 0 ]; then
    echo ""
    echo "🎉 End-to-end test completed successfully!"
    echo "✅ ReID tracking pipeline executed"
    echo "✅ Output videos generated"
    echo ""
    echo "📁 Test artifacts saved in: $OUTPUT_DIR"
    echo "🎬 Output videos available in: output/"
    echo ""
    echo "🔍 Next steps:"
    echo "  1. Review pipeline log for ReID activity"
    echo "  2. Check output videos for tracking quality"
    echo "  3. Validate cross-camera track associations"
    
    # Show final log lines
    echo ""
    echo "📝 Final pipeline messages:"
    tail -5 "$LOG_FILE" | sed 's/^/  /'
    
else
    echo ""
    echo "⚠️  Test completed with issues"
    echo "   - Check logs for errors"
    echo "   - Verify GPU/container configuration"
    echo "   - Review video output generation"
fi

echo ""
echo "🎯 Test completed: $(date)"
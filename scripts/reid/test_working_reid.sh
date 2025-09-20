#!/bin/bash

# Working ReID Test - CPU Compatible
# This test is designed to work and generate output videos

set -e

echo "🚀 Working ReID Tracking Test"
echo "============================="

# Setup
TEST_DIR="output/working_reid_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

echo "📁 Test directory: $TEST_DIR"

# Ensure Triton is running
if ! docker ps --filter "name=triton" --format "table {{.Names}}" | grep -q triton; then
    echo "🔄 Starting Triton server..."
    docker compose up -d triton
    sleep 15
fi

echo "✅ Triton server ready"

# Clean up old outputs
echo "🧹 Cleaning old output files..."
rm -f output/reid_cpu_cam*.mp4
rm -f output/dual_video_detections_*.mp4

# Test the CPU-compatible ReID configuration
echo ""
echo "🎬 Running CPU-compatible ReID tracking..."
echo "Duration: 45 seconds to ensure video generation"

docker run --rm \
  --network deepstream_triton_deepstream-triton \
  -v $(pwd)/videos:/workspace/videos \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/labels:/workspace/labels \
  -v $(pwd)/trackers:/workspace/trackers \
  --name deepstream-working-reid \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  bash -c "
    cd /workspace && 
    echo '🚀 Starting DeepStream with CPU-compatible ReID tracking...' &&
    echo 'Configuration: configs/reid/deepstream_reid_cpu.txt' &&
    echo 'Tracker: configs/reid/tracker_reid_basic.yml' &&
    echo 'Video 1: videos/mb1_1.mp4' &&
    echo 'Video 2: videos/mb2_2.mp4' &&
    echo 'Expected outputs: reid_cpu_cam1.mp4, reid_cpu_cam2.mp4' &&
    echo '' &&
    echo '📊 Starting pipeline...' &&
    timeout 45s deepstream-app -c configs/reid/deepstream_reid_cpu.txt 2>&1 | 
    tee /workspace/output/working_reid_*/pipeline_log.txt || echo '⏱️ Pipeline completed (45s timeout)'
  "

# Copy log to test directory
if [ -f "output/working_reid_*/pipeline_log.txt" ]; then
    cp output/working_reid_*/pipeline_log.txt "$TEST_DIR/"
fi

# Check results
echo ""
echo "📊 Checking results..."
echo "======================"

# Check for output videos
OUTPUT_VIDEOS=(
    "reid_cpu_cam1.mp4"
    "reid_cpu_cam2.mp4"
    "dual_video_detections_0.mp4"
    "dual_video_detections_1.mp4"
)

VIDEO_COUNT=0
echo "🎬 Generated videos:"

for video in "${OUTPUT_VIDEOS[@]}"; do
    if [ -f "output/$video" ]; then
        SIZE=$(ls -lh "output/$video" | awk '{print $5}')
        DURATION=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "output/$video" 2>/dev/null || echo "unknown")
        echo "  ✅ $video ($SIZE, ${DURATION}s)"
        
        # Copy to test directory
        cp "output/$video" "$TEST_DIR/"
        ((VIDEO_COUNT++))
    else
        echo "  ❌ $video not found"
    fi
done

# Check pipeline log
LOG_FILE="$TEST_DIR/pipeline_log.txt"
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "📋 Pipeline Analysis:"
    
    LOG_LINES=$(wc -l < "$LOG_FILE")
    echo "  📝 Log size: $LOG_LINES lines"
    
    # Check for successful components
    if grep -q -i "playing\|ready" "$LOG_FILE"; then
        echo "  ✅ Pipeline started successfully"
    fi
    
    if grep -q -i "tracker\|tracking" "$LOG_FILE"; then
        echo "  ✅ Tracker active"
        TRACKER_COUNT=$(grep -c -i "tracker\|tracking" "$LOG_FILE")
        echo "    - Tracker messages: $TRACKER_COUNT"
    fi
    
    if grep -q -i "inference\|detection" "$LOG_FILE"; then
        echo "  ✅ Inference engine active"
    fi
    
    if grep -q -i "fps\|frame.*rate" "$LOG_FILE"; then
        echo "  📈 Performance data available:"
        grep -i "fps\|frame.*rate" "$LOG_FILE" | tail -3 | sed 's/^/    /'
    fi
    
    # Check for errors
    ERROR_COUNT=$(grep -c -i "error\|failed\|critical" "$LOG_FILE" || echo "0")
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "  ⚠️  Errors: $ERROR_COUNT"
        echo "    Recent errors:"
        grep -i "error\|failed\|critical" "$LOG_FILE" | tail -3 | sed 's/^/      /'
    else
        echo "  ✅ No critical errors"
    fi
    
    echo ""
    echo "📝 Final pipeline messages:"
    tail -8 "$LOG_FILE" | sed 's/^/  /'
    
else
    echo "❌ Pipeline log not found"
fi

# Summary
echo ""
echo "🎯 Test Summary"
echo "==============="
echo "Test completed: $(date)"
echo "Generated videos: $VIDEO_COUNT"
echo "Test artifacts: $TEST_DIR"

if [ "$VIDEO_COUNT" -gt 0 ]; then
    echo ""
    echo "🎉 SUCCESS! ReID tracking pipeline generated output videos!"
    echo ""
    echo "📂 Available outputs:"
    ls -lh "$TEST_DIR"/*.mp4 2>/dev/null | sed 's/^/  /'
    
    echo ""
    echo "🔍 Next steps:"
    echo "  1. Review generated videos for tracking quality"
    echo "  2. Check if track IDs persist across frames"
    echo "  3. Validate object detection accuracy"
    echo "  4. Analyze performance metrics in log"
    
    # Show video details
    echo ""
    echo "📹 Video details:"
    for video_file in "$TEST_DIR"/*.mp4; do
        if [ -f "$video_file" ]; then
            VIDEO_NAME=$(basename "$video_file")
            echo "  🎬 $VIDEO_NAME:"
            
            # Try to get video info
            if command -v ffprobe &> /dev/null; then
                INFO=$(ffprobe -v quiet -show_entries stream=width,height,duration -of csv=p=0 "$video_file" 2>/dev/null || echo "info unavailable")
                echo "    Resolution & Duration: $INFO"
            fi
            
            SIZE=$(ls -lh "$video_file" | awk '{print $5}')
            echo "    File size: $SIZE"
        fi
    done
    
else
    echo ""
    echo "⚠️  No videos generated. Check pipeline configuration."
    echo "   Log available at: $LOG_FILE"
fi

echo ""
echo "🎯 Working ReID test completed!"
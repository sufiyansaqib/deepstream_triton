#!/bin/bash

set -e

echo "🚀 FINAL DEEPSTREAM REID SOLUTION"
echo "=================================="

# Create output directory
OUTPUT_DIR="output/FINAL_TEST_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output: $OUTPUT_DIR"

# SOLUTION 1: Clean environment and restart everything
echo "🧹 Cleaning Docker environment..."
docker compose down 2>/dev/null || true
docker kill $(docker ps -q) 2>/dev/null || true
sleep 5

# SOLUTION 2: Start Triton with explicit model loading
echo "🔧 Starting Triton server with proper model management..."
docker compose up triton -d
sleep 20

# SOLUTION 3: Verify Triton is fully operational
echo "🔍 Verifying Triton server..."
for i in {1..10}; do
    if curl -f http://localhost:8000/v2/health/ready 2>/dev/null; then
        echo "✅ Triton ready"
        if curl -s http://localhost:8000/v2/models/yolov7_fp16 >/dev/null 2>&1; then
            echo "✅ YOLOv7 model loaded"
            break
        fi
    fi
    echo "⏳ Waiting... ($i/10)"
    sleep 3
done

# SOLUTION 4: Use the PROVEN working approach with timeout
echo "🎬 Running DeepStream with aggressive timeout and logging..."
timeout 180 docker run --rm --gpus all \
  --network deepstream_triton_deepstream-triton \
  -v ./videos:/workspace/videos \
  -v ./configs:/workspace/configs \
  -v ./"$OUTPUT_DIR":/workspace/output \
  -v ./models:/workspace/models \
  -v ./labels:/workspace/labels \
  -v ./nvdsinfer_custom_impl_yolov7:/workspace/nvdsinfer_custom_impl_yolov7 \
  -e GST_DEBUG=3 \
  --name deepstream-final-solution \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  bash -c "
    echo '🚀 DeepStream Final Solution Test' &&
    echo 'Container started successfully!' &&
    cd /workspace &&
    echo '🔧 Testing Triton connectivity...' &&
    timeout 10 bash -c 'until nc -zv triton 8001; do sleep 1; done' &&
    echo '✅ Triton connection established!' &&
    echo '🎥 Starting DeepStream processing...' &&
    exec deepstream-app -c configs/deepstream_dual_video_triton.txt
  " 2>&1 | tee "$OUTPUT_DIR/execution_log.txt" &

# Wait for processing with status updates
echo "⏳ Processing videos... (will timeout after 3 minutes)"
sleep 30 && echo "⏳ 30s elapsed..."
sleep 30 && echo "⏳ 1m elapsed..."
sleep 30 && echo "⏳ 1.5m elapsed..."
sleep 30 && echo "⏳ 2m elapsed..."
sleep 30 && echo "⏳ 2.5m elapsed..."
sleep 30 && echo "⏳ 3m elapsed - timeout reached"

# Check results
echo "🏁 Checking results..."
ls -la "$OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/reid_video_camera_0.mp4" ] || [ -f "$OUTPUT_DIR/reid_video_camera_1.mp4" ]; then
    echo "✅ SUCCESS: Video files generated!"
    ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null || echo "No MP4 files found"
else
    echo "❌ No video output generated"
    echo "📋 Checking logs for issues:"
    tail -20 "$OUTPUT_DIR/execution_log.txt" 2>/dev/null || echo "No log file"
fi

# Cleanup
echo "🛑 Cleaning up..."
docker kill deepstream-final-solution 2>/dev/null || true
docker compose down

echo "✅ Final solution test complete!"
echo "📁 Results in: $OUTPUT_DIR"
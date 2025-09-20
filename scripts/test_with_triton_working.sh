#!/bin/bash

set -e

echo "🚀 Testing Complete DeepStream + Triton Pipeline (PROPER FIX)..."

# Create output directory with timestamp
OUTPUT_DIR="output/working_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"

# Step 1: Start Triton server FIRST
echo "🔧 Starting Triton server..."
docker compose up triton -d

# Step 2: Wait properly for Triton to be fully ready
echo "⏳ Waiting for Triton server to be fully ready..."
sleep 30

# Step 3: Test Triton connectivity thoroughly
echo "🔍 Testing Triton connectivity..."
for i in {1..20}; do
    if curl -f http://localhost:8000/v2/health/ready 2>/dev/null; then
        echo "✅ Triton server ready!"
        echo "📋 Available models:"
        curl -s http://localhost:8000/v2/models | jq '.[]' || echo "Models listed"
        echo "🎯 Testing yolov7_fp16 model status:"
        curl -s http://localhost:8000/v2/models/yolov7_fp16/ready || echo "Model status checked"
        break
    else
        echo "⏳ Waiting for Triton... ($i/20)"
        sleep 3
    fi
done

# Step 4: Run DeepStream with proper network and debugging
echo "🎬 Starting DeepStream with Triton connectivity..."
docker run --rm --gpus all \
  --network deepstream_triton_deepstream-triton \
  -v ./videos:/workspace/videos \
  -v ./configs:/workspace/configs \
  -v ./"$OUTPUT_DIR":/workspace/output \
  -v ./models:/workspace/models \
  -v ./labels:/workspace/labels \
  -v ./trackers:/workspace/trackers \
  -v ./nvdsinfer_custom_impl_yolov7:/workspace/nvdsinfer_custom_impl_yolov7 \
  -e GST_DEBUG=2 \
  -e TRITON_SERVER_URL=triton:8001 \
  --name deepstream-final-test \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  timeout 120 bash -c "
    cd /workspace && 
    echo '🔧 Testing network connectivity to Triton...' &&
    ping -c 3 triton || echo 'Ping failed but continuing...' &&
    echo '🔍 Testing Triton gRPC connection...' &&
    nc -zv triton 8001 && echo '✅ gRPC connection successful!' || echo '❌ gRPC connection failed' &&
    echo '🎥 Starting DeepStream application...' &&
    deepstream-app -c configs/deepstream_dual_video_triton.txt 2>&1 | 
    tee /workspace/output/final_test_log.txt
  "

echo "🏁 Final test completed! Results in: $OUTPUT_DIR"
echo "📄 Checking output files:"
ls -la "$OUTPUT_DIR"

# Step 5: Show logs for debugging
if [ -f "$OUTPUT_DIR/final_test_log.txt" ]; then
    echo "📋 Last 20 lines of log:"
    tail -20 "$OUTPUT_DIR/final_test_log.txt"
fi

# Step 6: Stop Triton
echo "🛑 Stopping Triton server..."
docker compose down

echo "✅ Complete pipeline test finished!"
#!/bin/bash

set -e

echo "🚀 Testing Simple Dual Video Pipeline (No ReID)..."

# Create output directory with timestamp
OUTPUT_DIR="output/simple_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"

# Step 1: Start Triton server with YOLOv7 only
echo "🔧 Starting Triton server with YOLOv7..."
docker compose up triton -d

# Step 2: Wait for Triton to be ready
echo "⏳ Waiting for Triton server to be ready..."
sleep 15

# Step 3: Test Triton connectivity
echo "🔍 Testing Triton connectivity..."
for i in {1..10}; do
    if curl -f http://localhost:8000/v2/health/ready 2>/dev/null; then
        echo "✅ Triton server ready!"
        curl -s http://localhost:8000/v2/models | jq '.' || echo "Models response received"
        break
    else
        echo "⏳ Waiting for Triton... ($i/10)"
        sleep 3
    fi
done

# Step 4: Test with working configuration
echo "🎬 Testing with working dual video configuration..."
docker run --rm --gpus all \
  --network deepstream_triton_deepstream-triton \
  -v ./videos:/workspace/videos \
  -v ./configs:/workspace/configs \
  -v ./"$OUTPUT_DIR":/workspace/output \
  -v ./models:/workspace/models \
  -v ./labels:/workspace/labels \
  -v ./trackers:/workspace/trackers \
  -e TRITON_SERVER_URL=triton:8001 \
  --name deepstream-simple-test \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  timeout 60 bash -c "
    cd /workspace && 
    echo '🔧 Testing Triton connection...' &&
    ping -c 2 triton || echo 'Ping failed but continuing...' &&
    echo '🎥 Starting DeepStream application...' &&
    deepstream-app -c configs/deepstream_dual_video_triton.txt 2>&1 | 
    tee /workspace/output/simple_test_log.txt
  "

echo "🏁 Simple test completed! Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"

# Step 5: Stop Triton
echo "🛑 Stopping Triton server..."
docker compose down

echo "✅ Simple dual video test complete!"
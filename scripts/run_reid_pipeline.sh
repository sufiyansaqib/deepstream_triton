#!/bin/bash

set -e

echo "🚀 Starting DeepStream ReID Pipeline Test..."

# Create output directory with timestamp
OUTPUT_DIR="output/reid_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"

# Step 1: Start Triton server with ReID models
echo "🔧 Starting Triton server with ReID models..."
docker compose -f docker-compose-reid.yml up triton-server -d

# Step 2: Wait for Triton to be ready
echo "⏳ Waiting for Triton server to be ready..."
sleep 20

# Step 3: Test Triton connectivity
echo "🔍 Testing Triton connectivity..."
curl -f http://localhost:8000/v2/health/ready || {
    echo "❌ Triton server not ready"
    docker-compose -f docker-compose-reid.yml logs triton-server
    exit 1
}

echo "✅ Triton server ready! Available models:"
curl -s http://localhost:8000/v2/models | jq '.' || echo "Models response received"

# Step 4: Run DeepStream with ReID
echo "🎬 Starting DeepStream ReID pipeline..."
docker run --rm --gpus all \
  --network deepstream_triton_reid \
  -v ./videos:/workspace/videos \
  -v ./configs:/workspace/configs \
  -v ./output:/workspace/output \
  -v ./models:/workspace/models \
  -v ./labels:/workspace/labels \
  -v ./trackers:/workspace/trackers \
  -v ./scripts:/workspace/scripts \
  -v ./logs:/workspace/logs \
  --name deepstream-reid-test \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  bash -c "
    cd /workspace && 
    echo '🔧 Testing Triton connection from DeepStream container...' &&
    timeout 30 bash scripts/test_triton_connection.sh triton-reid-server 8001 &&
    echo '🎥 Starting DeepStream ReID application...' &&
    deepstream-app -c configs/deepstream_dual_video_triton_reid.txt 2>&1 | 
    tee /workspace/logs/deepstream_reid_$(date +%Y%m%d_%H%M%S).log
  "

echo "🏁 Pipeline completed! Check output directory: $OUTPUT_DIR"
echo "📄 Logs available in: logs/"

# Step 5: Stop Triton server
echo "🛑 Stopping Triton server..."
docker compose -f docker-compose-reid.yml down

echo "✅ ReID pipeline test complete!"
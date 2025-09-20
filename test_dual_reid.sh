#!/bin/bash

echo "🎬 Testing Dual Video ReID Processing"
echo "===================================="

# Clean previous outputs
rm -f output/reid_video_detections_*.mp4

# Ensure Triton is running
echo "✅ Triton server status:"
docker ps --filter "name=triton" --format "table {{.Names}}\t{{.Status}}"

echo ""
echo "🚀 Starting DeepStream dual video processing..."
echo "Using exact working GPU setup"

# Use exact same approach as your working script but with ReID config
docker run --rm --gpus all \
  --network deepstream_triton_deepstream-triton \
  -v ./videos:/workspace/videos \
  -v ./configs:/workspace/configs \
  -v ./output:/workspace/output \
  -v ./models:/workspace/models \
  -v ./labels:/workspace/labels \
  -v ./trackers:/workspace/trackers \
  --name deepstream-reid-test \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  bash -c "
    cd /workspace && 
    echo 'DeepStream ReID application starting...' &&
    deepstream-app -c configs/reid/deepstream_reid_enabled.txt 2>&1 | 
    tee /workspace/output/reid_pipeline_log.txt
  "

echo ""
echo "📊 Checking results..."
if [ -f "output/reid_video_detections_0.mp4" ]; then
    echo "✅ Camera 0 output: $(ls -lh output/reid_video_detections_0.mp4 | awk '{print $5}')"
else
    echo "❌ Camera 0 output not found"
fi

if [ -f "output/reid_video_detections_1.mp4" ]; then
    echo "✅ Camera 1 output: $(ls -lh output/reid_video_detections_1.mp4 | awk '{print $5}')"
else
    echo "❌ Camera 1 output not found"
fi

if [ -f "output/reid_pipeline_log.txt" ]; then
    echo ""
    echo "📋 Pipeline log (last 20 lines):"
    tail -20 output/reid_pipeline_log.txt
fi

echo ""
echo "🎯 Test completed!"
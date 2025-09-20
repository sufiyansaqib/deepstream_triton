#!/bin/bash

echo "Starting DeepStream-Triton pipeline with FPS monitoring..."
echo "========================================================="

# Start Triton server
docker compose up -d triton

# Wait for Triton to be ready
echo "Waiting for Triton server to be ready..."
sleep 10

# Run DeepStream with logging
echo "Starting DeepStream application with performance monitoring..."
docker run --rm --gpus all \
  --network deepstream_triton_deepstream-triton \
  -v ./videos:/workspace/videos \
  -v ./configs:/workspace/configs \
  -v ./output:/workspace/output \
  -v ./models:/workspace/models \
  -v ./labels:/workspace/labels \
  --name deepstream-fps-monitor \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
  bash -c "
    cd /workspace && 
    echo 'DeepStream application starting...' &&
    deepstream-app -c configs/deepstream_dual_video_triton.txt 2>&1 | 
    tee /workspace/output/full_pipeline_log.txt
  "

echo "Pipeline execution completed!"
echo "Checking output files..."
ls -la output/
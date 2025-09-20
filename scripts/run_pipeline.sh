#!/bin/bash

# DeepStream Triton Pipeline Runner
# Starts the complete video processing pipeline

set -e

echo "=== Starting DeepStream Triton Pipeline ==="

# Check if setup was run
if [[ ! -f "models/yolov7_fp16/1/model.plan" ]]; then
    echo "ERROR: Model file not found. Please run ./scripts/setup.sh first"
    exit 1
fi

# Set X11 forwarding for display (if needed)
if [[ -z "${DISPLAY}" ]]; then
    export DISPLAY=:0
fi

# Allow X11 connections (for local display)
if command -v xhost &> /dev/null; then
    xhost +local:docker 2>/dev/null || echo "Note: Could not set X11 permissions"
fi

# Start the pipeline
echo "Starting Triton Inference Server..."
echo "Starting DeepStream Application..."
echo ""
echo "Pipeline will process videos from the videos/ directory"
echo "Output will be saved to the output/ directory"
echo "Press Ctrl+C to stop the pipeline"
echo ""

# Run with Docker Compose
docker compose up --build

echo ""
echo "Pipeline stopped. Check output/ directory for results."
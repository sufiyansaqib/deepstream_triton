#!/bin/bash

# DeepStream Triton Pipeline Setup Script
# Prepares environment and validates setup

set -e

echo "=== DeepStream Triton Pipeline Setup ==="

# Check if running with GPU support
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA GPU drivers not found. Please install NVIDIA drivers and Docker GPU support."
    exit 1
fi

# Check Docker Compose
if ! docker compose version &> /dev/null; then
    echo "ERROR: Docker Compose not found. Please install Docker Compose."
    exit 1
fi

# Validate model file exists
MODEL_FILE="models/yolov7_fp16/1/model.plan"
if [[ ! -f "$MODEL_FILE" ]]; then
    echo "ERROR: TensorRT model file not found at $MODEL_FILE"
    echo "Please copy your YOLOv7 TensorRT FP16 model to this location."
    exit 1
fi

# Validate video files
VIDEO_DIR="videos"
if [[ ! -d "$VIDEO_DIR" ]] || [[ -z "$(ls -A $VIDEO_DIR 2>/dev/null)" ]]; then
    echo "WARNING: No videos found in $VIDEO_DIR directory"
    echo "Please add video files (.mp4, .avi, etc.) to the videos/ directory"
    
    # Create sample video placeholders if none exist
    mkdir -p "$VIDEO_DIR"
    echo "Creating placeholder video references..."
    echo "# Add your video files here" > "$VIDEO_DIR/README.md"
fi

# Create output directory
mkdir -p output

# Set proper permissions
chmod +x scripts/*.sh

# Validate Triton model configuration
echo "Validating Triton model configuration..."
TRITON_CONFIG="models/yolov7_fp16/config.pbtxt"
if [[ ! -f "$TRITON_CONFIG" ]]; then
    echo "ERROR: Triton model configuration not found at $TRITON_CONFIG"
    exit 1
fi

echo "✓ GPU drivers detected"
echo "✓ Docker Compose available"
echo "✓ Model file present"
echo "✓ Configuration files validated"
echo ""
echo "Setup complete! You can now run:"
echo "  ./scripts/run_pipeline.sh     - Start the complete pipeline"
echo "  ./scripts/monitor.sh          - Monitor pipeline performance"
echo "  ./scripts/cleanup.sh          - Clean up containers and outputs"
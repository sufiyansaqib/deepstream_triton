#!/bin/bash

# Build script for marcoslucianops/DeepStream-Yolo parser
# Builds the parser library inside DeepStream container for compatibility

set -e

echo "Building marcoslucianops/DeepStream-Yolo parser library..."

# Build the Docker image with the compiled parser
echo "Building Docker image with compiled parser..."
docker build -f Dockerfile.deepstream-yolo -t deepstream-yolo:latest .

# Extract the compiled library
echo "Extracting compiled library..."
docker run --rm -v $(pwd):/host deepstream-yolo:latest cp /workspace/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so /host/

# Verify the library was extracted
if [ -f "libnvdsinfer_custom_impl_Yolo.so" ]; then
    echo "✅ Parser library built successfully!"
    echo "   Library size: $(ls -lh libnvdsinfer_custom_impl_Yolo.so | awk '{print $5}')"
    echo "   Location: $(pwd)/libnvdsinfer_custom_impl_Yolo.so"
else
    echo "❌ Failed to extract parser library"
    exit 1
fi

echo "🎉 Build complete! You can now use the new parser with your DeepStream configurations."
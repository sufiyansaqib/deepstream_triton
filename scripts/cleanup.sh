#!/bin/bash

# DeepStream Triton Pipeline Cleanup
# Stops containers and optionally cleans up outputs

echo "=== DeepStream Triton Pipeline Cleanup ==="

# Stop containers
echo "Stopping containers..."
docker compose down

# Remove dangling containers and networks
echo "Cleaning up Docker resources..."
docker compose down --remove-orphans

# Optional: Clean up output files
read -p "Do you want to remove output files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing output files..."
    rm -rf output/*.mp4 output/*.avi 2>/dev/null || echo "No output files to remove"
    echo "Output files removed."
fi

# Optional: Clean up Docker images
read -p "Do you want to remove Docker images? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing Docker images..."
    docker compose down --rmi all
    echo "Docker images removed."
fi

echo "Cleanup complete!"
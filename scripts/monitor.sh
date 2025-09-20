#!/bin/bash

# DeepStream Triton Pipeline Monitor
# Monitors performance and health of the pipeline

echo "=== DeepStream Triton Pipeline Monitor ==="

# Check if containers are running
echo "Checking container status..."
docker compose ps

echo ""
echo "=== Triton Server Health ==="
if docker compose exec -T triton curl -s http://localhost:8000/v2/health/ready | grep -q "ready"; then
    echo "✓ Triton server is ready"
else
    echo "✗ Triton server not ready"
fi

echo ""
echo "=== Triton Model Status ==="
docker compose exec -T triton curl -s http://localhost:8000/v2/models/yolov7_fp16 | jq -r '.state // "Not available"' 2>/dev/null || echo "Model status check failed"

echo ""
echo "=== GPU Utilization ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
else
    echo "nvidia-smi not available"
fi

echo ""
echo "=== DeepStream Performance ==="
echo "Check DeepStream logs for performance metrics:"
echo "docker compose logs deepstream | grep -E '(fps|Performance|Total)'"

echo ""
echo "=== Container Resource Usage ==="
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" $(docker compose ps -q) 2>/dev/null || echo "Container stats not available"

echo ""
echo "=== Recent Logs ==="
echo "Triton Server (last 10 lines):"
docker compose logs --tail=10 triton 2>/dev/null || echo "Triton logs not available"
echo ""
echo "DeepStream App (last 10 lines):"
docker compose logs --tail=10 deepstream 2>/dev/null || echo "DeepStream logs not available"
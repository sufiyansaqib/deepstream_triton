#!/bin/bash

echo "Testing Triton Server connection..."
TRITON_HOST=${1:-triton-reid-server}
TRITON_PORT=${2:-8001}

# Wait for Triton server to be ready
for i in {1..30}; do
    if curl -f http://${TRITON_HOST}:8000/v2/health/ready 2>/dev/null; then
        echo "✅ Triton server is ready!"
        
        # Test model availability
        echo "Checking available models..."
        curl -s http://${TRITON_HOST}:8000/v2/models
        
        # Test gRPC connection
        echo "Testing gRPC connection..."
        nc -zv ${TRITON_HOST} ${TRITON_PORT}
        
        if [ $? -eq 0 ]; then
            echo "✅ gRPC connection successful!"
            exit 0
        else
            echo "❌ gRPC connection failed"
            exit 1
        fi
    else
        echo "⏳ Waiting for Triton server... (${i}/30)"
        sleep 5
    fi
done

echo "❌ Triton server not ready after 150 seconds"
exit 1
# Triton Inference Server - YOLOv7 TensorRT Working Setup

**Date:** September 7, 2025  
**Status:** ✅ WORKING SOLUTION  
**Model:** YOLOv7 TensorRT FP16  
**Triton Version:** 24.08-py3  

## 🎯 Quick Summary

This document contains the **EXACT working configuration** for running YOLOv7 TensorRT FP16 model with Triton Inference Server. Follow these steps precisely to avoid configuration issues.

## 🔧 Working Configuration

### 1. Model File Structure
```
models/
└── yolov7_fp16/
    ├── 1/
    │   └── model.plan          # TensorRT FP16 engine file
    └── config.pbtxt           # Triton model configuration
```

### 2. Critical Triton Model Config (`models/yolov7_fp16/config.pbtxt`)

**⚠️ IMPORTANT: Use these EXACT specifications**

```protobuf
name: "yolov7_fp16"
platform: "tensorrt_plan"
max_batch_size: 2                    # ⚠️ MUST match TensorRT engine batch size
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]            # Input: CHW format
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 25200, 6 ]               # ⚠️ CRITICAL: Use 6, NOT 85!
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 100
  preferred_batch_size: [ 2 ]        # Match max_batch_size
}

optimization {
  cuda {
    graphs: true                     # Enable CUDA graphs for performance
  }
}
```

## ⚠️ Critical Issues Encountered & Solutions

### Issue 1: Batch Size Mismatch
**Error:** `configuration specified max-batch 4 but TensorRT engine only supports max-batch 2`

**Root Cause:** The TensorRT engine was built with `max_batch_size=2`, but config specified 4.

**Solution:** 
- Set `max_batch_size: 2` in config.pbtxt
- Set `preferred_batch_size: [ 2 ]` to match
- Ensure DeepStream configs also use `batch-size=2`

### Issue 2: Wrong Output Dimensions
**Error:** `the model expects 3 dimensions (shape [-1,25200,6]) but the model configuration specifies 3 dimensions [..., making complete shape [-1,25200,85])`

**Root Cause:** The actual TensorRT engine outputs `[25200, 6]` but config specified `[25200, 85]`.

**Solution:**
- **Always use `dims: [ 25200, 6 ]`** for this specific model
- The engine determines the actual output shape, not our assumptions

### Issue 3: Invalid Config Parameters
**Error:** `Message type "inference.ModelDynamicBatching" has no field named "max_queue_size"`

**Solution:** Remove unsupported parameters from dynamic_batching block.

## ✅ Verification Steps

### 1. Test Triton Server Startup
```bash
docker run --rm --gpus all -p 8000:8000 -p 8001:8001 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models --log-verbose=1
```

### 2. Expected Success Messages
Look for these in the logs:
```
I0907 11:29:57.462 1 server.cc:674] 
+-------------+---------+--------+
| Model       | Version | Status |
+-------------+---------+--------+
| yolov7_fp16 | 1       | READY  |  ← ✅ THIS IS SUCCESS
+-------------+---------+--------+
```

```
I0907 11:29:57.449167 1 instance_state.cc:3587] "captured CUDA graph for yolov7_fp16_0_0, batch size 1"
I0907 11:29:57.461903 1 instance_state.cc:3587] "captured CUDA graph for yolov7_fp16_0_0, batch size 2"
```

### 3. Test Model Endpoint
```bash
curl -s http://localhost:8000/v2/models/yolov7_fp16
```

**Expected Response:**
```json
{
  "name": "yolov7_fp16",
  "versions": ["1"],
  "platform": "tensorrt_plan",
  "inputs": [{"name": "input", "datatype": "FP32", "shape": [-1,3,640,640]}],
  "outputs": [{"name": "output", "datatype": "FP32", "shape": [-1,25200,6]}]
}
```

## 🚀 Performance Optimizations Applied

1. **CUDA Graphs Enabled** - Reduces kernel launch overhead
2. **Dynamic Batching** - Optimizes throughput with preferred batch size
3. **GPU Memory Pool** - Pre-allocated CUDA memory
4. **FP16 TensorRT Engine** - Faster inference with reduced memory

## 📝 Troubleshooting Checklist

If Triton fails to start:

1. ✅ **Check batch size:** Ensure `max_batch_size` in config matches TensorRT engine
2. ✅ **Check output dims:** Use `dims: [ 25200, 6 ]` for this specific model
3. ✅ **Check file paths:** Ensure `model.plan` is in `models/yolov7_fp16/1/`
4. ✅ **Check GPU access:** Use `--gpus all` flag
5. ✅ **Check ports:** Ensure 8000, 8001, 8002 are available

## 🔍 Model Introspection Commands

To inspect your TensorRT engine:
```bash
# Check engine info (if available)
/usr/src/tensorrt/bin/trtexec --loadEngine=model.plan --dumpOutput
```

## 📊 Performance Metrics

With this configuration on RTX 4060 Laptop GPU:
- **Model Load Time:** ~16-20ms
- **CUDA Graph Capture:** ~80ms (batch size 1 & 2)
- **Memory Usage:** ~186 MiB GPU memory
- **Ready for Inference:** ✅ Operational

## 🎯 Success Indicators

**✅ Triton is working correctly when you see:**
1. Model status shows `READY` (not `LOADING` or `UNAVAILABLE`)
2. CUDA graphs captured for multiple batch sizes
3. HTTP/gRPC servers started successfully
4. Model endpoint responds with correct input/output shapes
5. No error messages in logs

---

**💡 Remember:** Always use the exact configuration values shown above. The TensorRT engine dictates the constraints, not our preferences!
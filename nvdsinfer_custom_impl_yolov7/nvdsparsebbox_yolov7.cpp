/*
 * Custom YOLO v7 Triton Output Parsing for DeepStream
 * Adapted from NVIDIA DeepStream-Yolo implementation
 * 
 * Expected output format: [25200, 6] where each detection is:
 * [x1, y1, x2, y2, confidence, class_id]
 */

#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>

// Utility function to clamp values
static inline float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

// Convert raw bounding box coordinates to NvDsInferParseObjectInfo
static NvDsInferParseObjectInfo convertBBox(const float& bx1, const float& by1, const float& bx2, 
                                           const float& by2, const uint& netW, const uint& netH)
{
    NvDsInferParseObjectInfo b;
    
    // Clamp coordinates to network dimensions
    float x1 = clamp(bx1, 0, netW);
    float y1 = clamp(by1, 0, netH);
    float x2 = clamp(bx2, 0, netW);
    float y2 = clamp(by2, 0, netH);
    
    // Convert to DeepStream format (left, top, width, height)
    b.left = x1;
    b.width = clamp(x2 - x1, 0, netW);
    b.top = y1;
    b.height = clamp(y2 - y1, 0, netH);
    
    return b;
}

// Add bounding box proposal if it meets criteria
static void addBBoxProposal(const float bx1, const float by1, const float bx2, const float by2, 
                           const uint& netW, const uint& netH, const int maxIndex, const float maxProb, 
                           std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBox(bx1, by1, bx2, by2, netW, netH);
    
    // Skip invalid bounding boxes
    if (bbi.width < 1 || bbi.height < 1) {
        return;
    }
    
    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

// Decode raw YOLOv7 tensor output format (85 channels)
static std::vector<NvDsInferParseObjectInfo> decodeTensorYolov7Raw(const float* output, 
                                                                   const uint& outputSize, const uint& netW, 
                                                                   const uint& netH, const std::vector<float>& preclusterThreshold)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    
    // Process each detection in the raw output tensor
    // Raw YOLOv7 output format: [cx, cy, w, h, objectness, class0_prob, class1_prob, ..., class79_prob]
    for (uint b = 0; b < outputSize; ++b) {
        float objectness = output[b * 85 + 4];  // Object confidence
        
        if (objectness < 0.1) {  // Skip low objectness detections
            continue;
        }
        
        // Find best class
        float maxClassProb = 0.0f;
        int maxClassId = 0;
        for (int c = 0; c < 80; ++c) {
            float classProb = output[b * 85 + 5 + c];
            if (classProb > maxClassProb) {
                maxClassProb = classProb;
                maxClassId = c;
            }
        }
        
        // Calculate final confidence
        float confidence = objectness * maxClassProb;
        
        // Check if confidence meets threshold for this class
        if (maxClassId >= preclusterThreshold.size() || confidence < preclusterThreshold[maxClassId]) {
            continue;
        }
        
        // Extract and convert bounding box coordinates
        float cx = output[b * 85 + 0];  // Center x
        float cy = output[b * 85 + 1];  // Center y
        float w = output[b * 85 + 2];   // Width
        float h = output[b * 85 + 3];   // Height
        
        // Convert center coordinates to corner coordinates
        float bx1 = cx - w * 0.5f;
        float by1 = cy - h * 0.5f;
        float bx2 = cx + w * 0.5f;
        float by2 = cy + h * 0.5f;
        
        addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxClassId, confidence, binfo);
    }
    
    return binfo;
}

// Decode YOLOv7 tensor output format (6 channels - processed)
static std::vector<NvDsInferParseObjectInfo> decodeTensorYolov7(const float* output, 
                                                                const uint& outputSize, const uint& netW, 
                                                                const uint& netH, const std::vector<float>& preclusterThreshold)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    
    // Process each detection in the output tensor
    // YOLOv7 output format: [x1, y1, x2, y2, confidence, class_id]
    for (uint b = 0; b < outputSize; ++b) {
        float confidence = output[b * 6 + 4];  // Detection confidence
        int classId = (int)output[b * 6 + 5];  // Class ID
        
        // Check if confidence meets threshold for this class
        if (classId < 0 || classId >= preclusterThreshold.size()) {
            continue;
        }
        
        if (confidence < preclusterThreshold[classId]) {
            continue;
        }
        
        // Extract bounding box coordinates (already in x1, y1, x2, y2 format)
        float bx1 = output[b * 6 + 0];
        float by1 = output[b * 6 + 1]; 
        float bx2 = output[b * 6 + 2];
        float by2 = output[b * 6 + 3];
        
        addBBoxProposal(bx1, by1, bx2, by2, netW, netH, classId, confidence, binfo);
    }
    
    return binfo;
}

// Main parsing function for YOLOv7 Triton output
static bool NvDsInferParseCustomYolov7(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                       NvDsInferNetworkInfo const& networkInfo, 
                                       NvDsInferParseDetectionParams const& detectionParams,
                                       std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
        return false;
    }
    
    std::vector<NvDsInferParseObjectInfo> objects;
    
    const NvDsInferLayerInfo& output = outputLayersInfo[0];
    
    // Debug output dimensions
    std::cout << "YOLOv7 Output Tensor Debug Info:" << std::endl;
    std::cout << "  Number of dimensions: " << output.inferDims.numDims << std::endl;
    for (int i = 0; i < output.inferDims.numDims; i++) {
        std::cout << "  Dimension[" << i << "]: " << output.inferDims.d[i] << std::endl;
    }
    
    // Handle different output formats
    uint outputSize, outputChannels;
    
    if (output.inferDims.numDims == 2) {
        // Format: [num_detections, 6] - Direct DeepStream format
        outputSize = output.inferDims.d[0];
        outputChannels = output.inferDims.d[1];
    } else if (output.inferDims.numDims == 3) {
        // Format: [1, num_detections, 6] - Batch format
        outputSize = output.inferDims.d[1];
        outputChannels = output.inferDims.d[2];
        std::cout << "  Using batch format, batch_size=" << output.inferDims.d[0] << std::endl;
    } else {
        std::cerr << "ERROR: YOLOv7 output should have 2 or 3 dimensions, got: " 
                  << output.inferDims.numDims << std::endl;
        return false;
    }
    
    std::cout << "YOLOv7 Parsed Dimensions: size=" << outputSize << ", channels=" << outputChannels << std::endl;
    
    if (outputChannels != 6 && outputChannels != 85) {
        std::cerr << "ERROR: YOLOv7 output should have 6 channels [x1,y1,x2,y2,conf,class] or 85 channels [raw], got: " 
                  << outputChannels << std::endl;
        return false;
    }
    
    // Handle raw YOLOv7 output (85 channels) vs processed output (6 channels)
    if (outputChannels == 85) {
        std::cout << "WARNING: Raw YOLOv7 output detected (85 channels). Model needs DeepStreamOutput layer!" << std::endl;
        std::cout << "Using fallback parsing for raw output..." << std::endl;
    }
    
    // Decode detections from tensor using appropriate decoder
    std::vector<NvDsInferParseObjectInfo> outObjs;
    if (outputChannels == 85) {
        // Use raw YOLOv7 decoder
        outObjs = decodeTensorYolov7Raw((const float*)(output.buffer), 
                                        outputSize, networkInfo.width, 
                                        networkInfo.height, 
                                        detectionParams.perClassPreclusterThreshold);
    } else {
        // Use processed YOLOv7 decoder (6 channels)
        outObjs = decodeTensorYolov7((const float*)(output.buffer), 
                                     outputSize, networkInfo.width, 
                                     networkInfo.height, 
                                     detectionParams.perClassPreclusterThreshold);
    }
    
    objects.insert(objects.end(), outObjs.begin(), outObjs.end());
    
    std::cout << "YOLOv7 Parsed " << objects.size() << " objects from " << outputSize << " detections" << std::endl;
    
    objectList = objects;
    
    return true;
}

// External interface function
extern "C" bool NvDsInferParseYolov7(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, 
                                     NvDsInferNetworkInfo const& networkInfo,
                                     NvDsInferParseDetectionParams const& detectionParams, 
                                     std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseCustomYolov7(outputLayersInfo, networkInfo, detectionParams, objectList);
}

// Prototype check
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolov7);
"""
ReID Processor for DeepStream Multi-Camera Tracking
==================================================

Handles ReID feature extraction, processing, and integration with
DeepStream pipeline for cross-camera re-identification.
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import pyds
import numpy as np
import cv2
import time
import asyncio
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

from .global_track_manager import Detection, GlobalTrackManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReIDConfig:
    """ReID processing configuration"""
    feature_dim: int = 256
    max_batch_size: int = 100
    input_height: int = 256
    input_width: int = 128
    confidence_threshold: float = 0.5
    extraction_interval: int = 0  # 0 = every frame
    normalize_features: bool = True

class ReIDProcessor:
    """
    Processes ReID features from DeepStream metadata and coordinates
    with GlobalTrackManager for cross-camera tracking.
    """
    
    def __init__(self, 
                 global_manager: GlobalTrackManager,
                 config: ReIDConfig = None):
        """
        Initialize ReID Processor.
        
        Args:
            global_manager: GlobalTrackManager instance
            config: ReID configuration parameters
        """
        self.global_manager = global_manager
        self.config = config or ReIDConfig()
        
        # Processing statistics
        self.stats = {
            'frames_processed': 0,
            'detections_processed': 0,
            'reid_features_extracted': 0,
            'processing_times': [],
            'errors': 0
        }
        
        # Feature cache for performance
        self.feature_cache = {}
        self.cache_timeout = 5.0  # seconds
        
        logger.info("ReIDProcessor initialized")
    
    def create_gst_probe_callback(self):
        """
        Create GStreamer probe callback for ReID processing.
        
        Returns:
            Callback function for GStreamer pad probe
        """
        def reid_probe_callback(pad, info):
            """Process ReID features from DeepStream metadata"""
            try:
                start_time = time.time()
                
                # Get buffer and batch metadata
                buffer = info.get_buffer()
                if not buffer:
                    return Gst.PadProbeReturn.OK
                
                batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buffer)
                if not batch_meta:
                    return Gst.PadProbeReturn.OK
                
                # Process each frame in the batch
                frame_meta_list = batch_meta.frame_meta_list
                while frame_meta_list:
                    try:
                        frame_meta = pyds.NvDsFrameMeta.cast(frame_meta_list.data)
                        self._process_frame_metadata(frame_meta)
                        frame_meta_list = frame_meta_list.next
                    except StopIteration:
                        break
                
                # Update statistics
                self.stats['frames_processed'] += 1
                processing_time = (time.time() - start_time) * 1000
                self.stats['processing_times'].append(processing_time)
                
                # Keep only recent processing times
                if len(self.stats['processing_times']) > 1000:
                    self.stats['processing_times'] = self.stats['processing_times'][-1000:]
                
                return Gst.PadProbeReturn.OK
                
            except Exception as e:
                logger.error(f"Error in ReID probe callback: {e}")
                self.stats['errors'] += 1
                return Gst.PadProbeReturn.OK
        
        return reid_probe_callback
    
    def _process_frame_metadata(self, frame_meta):
        """Process metadata from a single frame"""
        camera_id = frame_meta.source_id
        frame_number = frame_meta.frame_num
        
        # Process each object in the frame
        obj_meta_list = frame_meta.obj_meta_list
        while obj_meta_list:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(obj_meta_list.data)
                
                # Skip low-confidence detections
                if obj_meta.confidence < self.config.confidence_threshold:
                    obj_meta_list = obj_meta_list.next
                    continue
                
                # Extract detection information
                detection = self._extract_detection_from_metadata(obj_meta, camera_id)
                
                if detection:
                    # Process asynchronously to avoid blocking pipeline
                    asyncio.create_task(
                        self.global_manager.associate_detection(detection)
                    )
                    self.stats['detections_processed'] += 1
                
                obj_meta_list = obj_meta_list.next
                
            except StopIteration:
                break
            except Exception as e:
                logger.warning(f"Error processing object metadata: {e}")
                try:
                    obj_meta_list = obj_meta_list.next
                except:
                    break
    
    def _extract_detection_from_metadata(self, obj_meta, camera_id: int) -> Optional[Detection]:
        """Extract detection information from DeepStream object metadata"""
        try:
            # Basic detection info
            detection = Detection(
                camera_id=camera_id,
                local_id=obj_meta.object_id,
                confidence=obj_meta.confidence,
                bbox=[
                    obj_meta.rect_params.left,
                    obj_meta.rect_params.top,
                    obj_meta.rect_params.width,
                    obj_meta.rect_params.height
                ],
                class_id=obj_meta.class_id,
                timestamp=time.time()
            )
            
            # Extract ReID features from user metadata
            reid_features = self._extract_reid_features(obj_meta)
            if reid_features is not None:
                detection.reid_features = reid_features
                self.stats['reid_features_extracted'] += 1
            
            return detection
            
        except Exception as e:
            logger.warning(f"Error extracting detection from metadata: {e}")
            return None
    
    def _extract_reid_features(self, obj_meta) -> Optional[np.ndarray]:
        """Extract ReID features from object metadata"""
        try:
            # Iterate through user metadata to find ReID features
            user_meta_list = obj_meta.obj_user_meta_list
            while user_meta_list:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                    
                    # Check for ReID tensor metadata
                    if (hasattr(user_meta, 'base_meta') and 
                        user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META):
                        
                        # Extract tensor data
                        tensor_meta = pyds.NvDsTensorMeta.cast(user_meta.user_meta_data)
                        if tensor_meta and tensor_meta.tensor_name == "reid_features":
                            
                            # Convert to numpy array
                            features = np.array(tensor_meta.tensor_data)
                            
                            # Reshape to expected dimensions
                            if len(features.shape) == 1 and features.shape[0] == self.config.feature_dim:
                                if self.config.normalize_features:
                                    features = self._normalize_features(features)
                                return features
                    
                    # Check for custom ReID metadata format
                    elif (hasattr(user_meta, 'user_meta_data') and 
                          hasattr(user_meta.user_meta_data, 'reid_feature_vector')):
                        
                        features = np.array(user_meta.user_meta_data.reid_feature_vector)
                        if self.config.normalize_features:
                            features = self._normalize_features(features)
                        return features
                    
                    user_meta_list = user_meta_list.next
                    
                except StopIteration:
                    break
                except Exception as e:
                    logger.debug(f"Error processing user metadata: {e}")
                    try:
                        user_meta_list = user_meta_list.next
                    except:
                        break
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting ReID features: {e}")
            return None
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature vector to unit length"""
        try:
            norm = np.linalg.norm(features)
            if norm > 1e-8:
                return features / norm
            else:
                return features
        except Exception as e:
            logger.warning(f"Error normalizing features: {e}")
            return features
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        avg_processing_time = (np.mean(self.stats['processing_times']) 
                             if self.stats['processing_times'] else 0)
        
        return {
            'frames_processed': self.stats['frames_processed'],
            'detections_processed': self.stats['detections_processed'],
            'reid_features_extracted': self.stats['reid_features_extracted'],
            'avg_processing_time_ms': avg_processing_time,
            'total_errors': self.stats['errors'],
            'reid_extraction_rate': (self.stats['reid_features_extracted'] / 
                                   max(1, self.stats['detections_processed'])),
            'cache_size': len(self.feature_cache)
        }
    
    def clear_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'frames_processed': 0,
            'detections_processed': 0,
            'reid_features_extracted': 0,
            'processing_times': [],
            'errors': 0
        }
        logger.info("ReID processing statistics cleared")
    
    def cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.feature_cache.items()
            if current_time - timestamp > self.cache_timeout
        ]
        
        for key in expired_keys:
            del self.feature_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
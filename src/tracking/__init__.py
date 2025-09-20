"""
Advanced Multi-Camera Tracking Module
====================================

This module provides advanced tracking capabilities including:
- Cross-camera re-identification (ReID)
- Global track management
- Multi-stream coordination
- Feature extraction and matching

Components:
-----------
- GlobalTrackManager: Manages tracks across multiple cameras
- ReIDProcessor: Handles re-identification feature processing
- CrossCameraAssociator: Associates tracks between cameras
- TrackingMetrics: Performance monitoring and validation
"""

from .global_track_manager import GlobalTrackManager
from .reid_processor import ReIDProcessor
from .cross_camera_associator import CrossCameraAssociator
from .tracking_metrics import TrackingMetrics

__version__ = "1.0.0"
__author__ = "DeepStream Advanced Tracking Team"

__all__ = [
    "GlobalTrackManager",
    "ReIDProcessor", 
    "CrossCameraAssociator",
    "TrackingMetrics"
]
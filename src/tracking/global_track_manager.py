"""
Global Track Manager for Multi-Camera Tracking
==============================================

Manages track associations across multiple camera streams with ReID-based
re-identification for robust cross-camera tracking.
"""

import numpy as np
import time
import asyncio
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.spatial.distance import cosine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Detection data structure"""
    camera_id: int
    local_id: int
    global_id: Optional[str] = None
    confidence: float = 0.0
    bbox: List[float] = None  # [x, y, w, h]
    reid_features: Optional[np.ndarray] = None
    timestamp: float = 0.0
    class_id: int = 0
    
    def __post_init__(self):
        if self.bbox is None:
            self.bbox = [0.0, 0.0, 0.0, 0.0]
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class GlobalTrack:
    """Global track data structure"""
    global_id: str
    cameras_seen: set
    last_seen: float
    reid_features: deque
    trajectory_history: Dict[int, List[Dict]]
    confidence_scores: List[float]
    creation_time: float
    total_detections: int = 0
    
    def __post_init__(self):
        if not isinstance(self.reid_features, deque):
            self.reid_features = deque(maxlen=100)
        if not isinstance(self.confidence_scores, list):
            self.confidence_scores = []

class GlobalTrackManager:
    """
    Manages global tracks across multiple camera streams.
    
    Features:
    - ReID-based cross-camera association
    - Temporal track management
    - Confidence-based track validation
    - Performance metrics tracking
    """
    
    def __init__(self, 
                 reid_threshold: float = 0.75,
                 max_history: int = 100,
                 track_timeout: float = 30.0,
                 min_confidence: float = 0.5):
        """
        Initialize Global Track Manager.
        
        Args:
            reid_threshold: Cosine similarity threshold for ReID matching
            max_history: Maximum feature history per track
            track_timeout: Time (seconds) before track is considered stale
            min_confidence: Minimum confidence for track creation
        """
        self.camera_tracks = defaultdict(dict)  # camera_id -> {local_id: global_id}
        self.global_tracks = {}  # global_id -> GlobalTrack
        self.reid_threshold = reid_threshold
        self.max_history = max_history
        self.track_timeout = track_timeout
        self.min_confidence = min_confidence
        self.global_id_counter = 0
        
        # Performance metrics
        self.metrics = {
            'total_detections': 0,
            'cross_camera_associations': 0,
            'new_tracks_created': 0,
            'tracks_timeout': 0,
            'processing_time_ms': []
        }
        
        logger.info(f"GlobalTrackManager initialized with threshold={reid_threshold}")
    
    def compute_reid_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute cosine similarity between ReID feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            # Handle different feature vector shapes
            if len(features1.shape) > 1:
                features1 = features1.flatten()
            if len(features2.shape) > 1:
                features2 = features2.flatten()
                
            # Normalize features
            features1 = features1 / (np.linalg.norm(features1) + 1e-8)
            features2 = features2 / (np.linalg.norm(features2) + 1e-8)
            
            # Compute cosine similarity
            similarity = 1 - cosine(features1, features2)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"Error computing ReID similarity: {e}")
            return 0.0
    
    async def associate_detection(self, detection: Detection) -> str:
        """
        Associate a new detection with existing global tracks.
        
        Args:
            detection: Detection object to associate
            
        Returns:
            Global track ID (existing or newly created)
        """
        start_time = time.time()
        self.metrics['total_detections'] += 1
        
        try:
            # Clean up stale tracks first
            await self._cleanup_stale_tracks()
            
            # Skip if detection confidence is too low
            if detection.confidence < self.min_confidence:
                logger.debug(f"Detection confidence {detection.confidence} below threshold")
                return self._create_new_global_track(detection)
            
            # If no ReID features available, create new track
            if detection.reid_features is None:
                logger.debug("No ReID features available for detection")
                return self._create_new_global_track(detection)
            
            # Find best matching track
            best_match_id, best_score = await self._find_best_match(detection)
            
            if best_match_id and best_score > self.reid_threshold:
                # Update existing track
                global_id = self._update_existing_track(best_match_id, detection)
                self.metrics['cross_camera_associations'] += 1
                logger.info(f"Associated detection with existing track {global_id} (score: {best_score:.3f})")
            else:
                # Create new track
                global_id = self._create_new_global_track(detection)
                logger.info(f"Created new global track {global_id}")
            
            # Update camera-local mapping
            self.camera_tracks[detection.camera_id][detection.local_id] = global_id
            
            # Record processing time
            processing_time = (time.time() - start_time) * 1000
            self.metrics['processing_time_ms'].append(processing_time)
            
            return global_id
            
        except Exception as e:
            logger.error(f"Error in associate_detection: {e}")
            return self._create_new_global_track(detection)
    
    async def _find_best_match(self, detection: Detection) -> Tuple[Optional[str], float]:
        """Find best matching global track for detection."""
        best_match_id = None
        best_score = 0.0
        
        # Compare with all existing global tracks
        for global_id, track in self.global_tracks.items():
            # Skip if same camera (within-camera tracking handled by local tracker)
            if detection.camera_id in track.cameras_seen:
                continue
            
            # Skip if track is too old
            if time.time() - track.last_seen > self.track_timeout:
                continue
                
            # Compare ReID features
            if track.reid_features and detection.reid_features is not None:
                # Use recent features for comparison
                recent_features = list(track.reid_features)[-10:]  # Last 10 features
                
                similarities = []
                for track_feature in recent_features:
                    sim = self.compute_reid_similarity(detection.reid_features, track_feature)
                    similarities.append(sim)
                
                if similarities:
                    # Use maximum similarity for robustness
                    max_similarity = max(similarities)
                    avg_similarity = np.mean(similarities)
                    
                    # Weighted score combining max and average
                    final_score = 0.7 * max_similarity + 0.3 * avg_similarity
                    
                    if final_score > best_score:
                        best_score = final_score
                        best_match_id = global_id
        
        return best_match_id, best_score
    
    def _create_new_global_track(self, detection: Detection) -> str:
        """Create a new global track for unmatched detection."""
        self.global_id_counter += 1
        global_id = f"GT_{self.global_id_counter:06d}"
        
        # Create global track
        global_track = GlobalTrack(
            global_id=global_id,
            cameras_seen={detection.camera_id},
            last_seen=detection.timestamp,
            reid_features=deque(maxlen=self.max_history),
            trajectory_history=defaultdict(list),
            confidence_scores=[detection.confidence],
            creation_time=detection.timestamp,
            total_detections=1
        )
        
        # Add ReID features if available
        if detection.reid_features is not None:
            global_track.reid_features.append(detection.reid_features.copy())
        
        # Add trajectory point
        global_track.trajectory_history[detection.camera_id].append({
            'timestamp': detection.timestamp,
            'bbox': detection.bbox.copy(),
            'confidence': detection.confidence,
            'local_id': detection.local_id
        })
        
        self.global_tracks[global_id] = global_track
        self.metrics['new_tracks_created'] += 1
        
        return global_id
    
    def _update_existing_track(self, global_id: str, detection: Detection) -> str:
        """Update existing global track with new detection."""
        track = self.global_tracks[global_id]
        
        # Update track metadata
        track.cameras_seen.add(detection.camera_id)
        track.last_seen = detection.timestamp
        track.total_detections += 1
        track.confidence_scores.append(detection.confidence)
        
        # Add ReID features
        if detection.reid_features is not None:
            track.reid_features.append(detection.reid_features.copy())
        
        # Add trajectory point
        track.trajectory_history[detection.camera_id].append({
            'timestamp': detection.timestamp,
            'bbox': detection.bbox.copy(),
            'confidence': detection.confidence,
            'local_id': detection.local_id
        })
        
        return global_id
    
    async def _cleanup_stale_tracks(self):
        """Remove tracks that haven't been seen for too long."""
        current_time = time.time()
        stale_tracks = []
        
        for global_id, track in self.global_tracks.items():
            if current_time - track.last_seen > self.track_timeout:
                stale_tracks.append(global_id)
        
        for global_id in stale_tracks:
            del self.global_tracks[global_id]
            self.metrics['tracks_timeout'] += 1
            logger.debug(f"Removed stale track {global_id}")
        
        # Clean up camera mappings
        for camera_id in self.camera_tracks:
            stale_locals = []
            for local_id, global_id in self.camera_tracks[camera_id].items():
                if global_id not in self.global_tracks:
                    stale_locals.append(local_id)
            
            for local_id in stale_locals:
                del self.camera_tracks[camera_id][local_id]
    
    def get_track_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics."""
        current_time = time.time()
        
        stats = {
            'total_global_tracks': len(self.global_tracks),
            'active_tracks': sum(1 for t in self.global_tracks.values() 
                               if current_time - t.last_seen < self.track_timeout),
            'cross_camera_tracks': sum(1 for t in self.global_tracks.values() 
                                     if len(t.cameras_seen) > 1),
            'cameras_tracked': len(self.camera_tracks),
            'avg_processing_time_ms': np.mean(self.metrics['processing_time_ms'][-1000:]) 
                                    if self.metrics['processing_time_ms'] else 0,
            'metrics': self.metrics.copy()
        }
        
        return stats
    
    def get_global_track(self, global_id: str) -> Optional[GlobalTrack]:
        """Get global track by ID."""
        return self.global_tracks.get(global_id)
    
    def get_camera_tracks(self, camera_id: int) -> Dict[int, str]:
        """Get all tracks for a specific camera."""
        return self.camera_tracks.get(camera_id, {}).copy()
    
    def export_tracks_for_validation(self) -> Dict[str, Any]:
        """Export tracks in format suitable for validation."""
        export_data = {
            'timestamp': time.time(),
            'tracks': {},
            'statistics': self.get_track_statistics()
        }
        
        for global_id, track in self.global_tracks.items():
            export_data['tracks'][global_id] = {
                'cameras_seen': list(track.cameras_seen),
                'total_detections': track.total_detections,
                'creation_time': track.creation_time,
                'last_seen': track.last_seen,
                'trajectory_points': sum(len(traj) for traj in track.trajectory_history.values()),
                'avg_confidence': np.mean(track.confidence_scores) if track.confidence_scores else 0
            }
        
        return export_data
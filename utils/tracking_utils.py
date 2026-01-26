"""
Object tracking utilities with SORT algorithm and Kalman filtering.
Provides consistent ID assignment across frames.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Optional
from loguru import logger


class KalmanBoxTracker:
    """Kalman filter for tracking bounding boxes in image space."""
    
    count = 0
    
    def __init__(self, bbox: np.ndarray):
        """
        Initialize tracker with bounding box.
        bbox: [x1, y1, x2, y2]
        """
        # State: [x, y, s, r, vx, vy, vs]
        # x, y: center coordinates
        # s: scale (area)
        # r: aspect ratio
        # v*: velocities
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        
        # Process covariance
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self._bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, bbox: np.ndarray):
        """Update tracker with new detection."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))
    
    def predict(self) -> np.ndarray:
        """Predict next state and return predicted bbox."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        self.history.append(self._z_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self) -> np.ndarray:
        """Return current bounding box estimate."""
        return self._z_to_bbox(self.kf.x)
    
    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """Convert [x1,y1,x2,y2] to [cx,cy,s,r]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / max(h, 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> np.ndarray:
        """Convert [cx,cy,s,r] to [x1,y1,x2,y2]."""
        w = np.sqrt(z[2] * z[3])
        h = z[2] / max(w, 1e-6)
        x1 = z[0] - w / 2.0
        y1 = z[1] - h / 2.0
        x2 = z[0] + w / 2.0
        y2 = z[1] + h / 2.0
        return np.array([x1, y1, x2, y2]).flatten()


class SORTTracker:
    """Simple Online and Realtime Tracking (SORT) algorithm."""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
    
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Update tracker with new detections.
        detections: [[x1,y1,x2,y2,score], ...]
        Returns: [[x1,y1,x2,y2,track_id], ...]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            # Ensure pos is a 1D array
            if np.isscalar(pos) or pos.ndim == 0:
                pos = np.array([pos])
            elif pos.ndim > 1:
                pos = pos.flatten()
            
            # Ensure we have at least 4 elements
            if len(pos) >= 4:
                trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Association using Hungarian algorithm
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, trks, self.iou_threshold
        )
        
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(detections[m[0], :4])
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i, :4])
            self.trackers.append(trk)
        
        # Return active tracks
        ret = []
        for trk in self.trackers:
            d = trk.get_state()  # Returns [x1, y1, x2, y2] - 1D array of 4 elements
            # Ensure d is properly shaped for concatenation
            if d.ndim == 0:
                d = np.array([d])
            elif d.ndim > 1:
                d = d.flatten()
            
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            
            # Remove dead tracklets
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    @staticmethod
    def _iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
        """Compute IoU between two bounding boxes."""
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        
        intersection = w * h
        area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        union = area_test + area_gt - intersection
        
        return intersection / max(union, 1e-6)
    
    def _associate_detections_to_trackers(
        self, detections: np.ndarray, trackers: np.ndarray, iou_threshold: float = 0.3
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """Assign detections to tracked objects using IoU."""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det[:4], trk[:4])
        
        # Hungarian algorithm
        if min(iou_matrix.shape) > 0:
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter matches with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, unmatched_detections, unmatched_trackers


class LandmarkKalmanFilter:
    """Kalman filter for smoothing pose landmarks (2D points)."""
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        # State: [x, y, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Noise
        self.kf.R *= measurement_noise
        self.kf.Q *= process_noise
        
        self.initialized = False
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update with new measurement [x, y]."""
        measurement = np.asarray(measurement).flatten()
        
        if not self.initialized:
            # Reshape measurement to match state vector shape
            self.kf.x[0, 0] = measurement[0]
            self.kf.x[1, 0] = measurement[1]
            self.initialized = True
        else:
            self.kf.predict()
            self.kf.update(measurement)
        
        # Ensure output is always a 1D array with shape (2,)
        position = np.asarray(self.kf.x[:2]).flatten()
        return position if position.shape == (2,) else np.array([position[0], position[1]], dtype=np.float32)
    
    def get_velocity(self) -> np.ndarray:
        """Get velocity estimate [vx, vy]."""
        # Ensure output is always a 1D array with shape (2,)
        velocity = np.asarray(self.kf.x[2:]).flatten()
        return velocity if velocity.shape == (2,) else np.array([velocity[0], velocity[1]], dtype=np.float32)

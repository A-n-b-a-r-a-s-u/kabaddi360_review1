"""
Stage 3: Pose Estimation (Raider Only)
MediaPipe pose estimation with Kalman filtering for smooth joint tracking.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from loguru import logger

try:
    import mediapipe as mp
except ImportError:
    logger.error("mediapipe not installed. Run: pip install mediapipe")
    raise

from utils.tracking_utils import LandmarkKalmanFilter
from config.config import POSE_CONFIG, KEY_JOINTS


class PoseEstimator:
    """Estimate pose for the raider using MediaPipe."""
    
    def __init__(self):
        """Initialize MediaPipe pose estimation."""
        logger.info("Initializing Pose Estimator...")
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=POSE_CONFIG["model_complexity"],
            smooth_landmarks=POSE_CONFIG["smooth_landmarks"],
            enable_segmentation=POSE_CONFIG["enable_segmentation"],
            min_detection_confidence=POSE_CONFIG["min_detection_confidence"],
            min_tracking_confidence=POSE_CONFIG["min_tracking_confidence"]
        )
        
        # Kalman filters for each joint
        self.joint_filters: Dict[str, LandmarkKalmanFilter] = {}
        self._initialize_filters()
        
        # Joint history
        self.joint_history: Dict[str, List[np.ndarray]] = {joint: [] for joint in KEY_JOINTS.keys()}
        self.max_history = 90  # 3 seconds at 30fps
        
        logger.info("Pose Estimator initialized successfully")
    
    def _initialize_filters(self):
        """Initialize Kalman filters for each key joint."""
        for joint_name in KEY_JOINTS.keys():
            try:
                self.joint_filters[joint_name] = LandmarkKalmanFilter(
                    process_noise=POSE_CONFIG["kalman_process_noise"],
                    measurement_noise=POSE_CONFIG["kalman_measurement_noise"]
                )
            except Exception as e:
                logger.warning(f"Failed to initialize filter for joint {joint_name}: {e}")
    
    def extract_pose(self, frame: np.ndarray, bbox: List[float]) -> Optional[Dict]:
        """
        Extract pose landmarks from raider bounding box.
        
        Args:
            frame: Full frame image
            bbox: Raider bounding box [x1, y1, x2, y2]
        
        Returns:
            Dictionary with joint positions and velocities, or None if detection fails
        """
        logger.debug(f"[STAGE 3 - POSE ESTIMATOR] Extracting pose from bbox: {bbox}")
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Expand bbox slightly for better pose detection
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        
        # Crop raider region
        raider_crop = frame[y1:y2, x1:x2]
        logger.debug(f"[STAGE 3 - POSE ESTIMATOR] Crop shape: {raider_crop.shape}")
        
        if raider_crop.size == 0:
            logger.warning(f"[STAGE 3 - POSE ESTIMATOR] Empty crop, pose extraction failed")
            return None
        
        # Convert to RGB for MediaPipe
        rgb_crop = cv2.cvtColor(raider_crop, cv2.COLOR_BGR2RGB)
        
        # Process pose
        results = self.pose.process(rgb_crop)
        
        if not results.pose_landmarks:
            logger.debug(f"[STAGE 3 - POSE ESTIMATOR] No pose landmarks detected")
            return None
        
        logger.debug(f"[STAGE 3 - POSE ESTIMATOR] Pose detected with {len(results.pose_landmarks.landmark)} landmarks")
        
        # Extract key joints
        joints = {}
        crop_height, crop_width = raider_crop.shape[:2]
        
        for joint_name, landmark_idx in KEY_JOINTS.items():
            if landmark_idx < len(results.pose_landmarks.landmark):
                landmark = results.pose_landmarks.landmark[landmark_idx]
                
                # Convert normalized coordinates to pixel coordinates
                x = landmark.x * crop_width + x1
                y = landmark.y * crop_height + y1
                
                # Apply Kalman filtering
                smoothed_pos = self.joint_filters[joint_name].update(np.array([x, y]))
                velocity = self.joint_filters[joint_name].get_velocity()
                
                joints[joint_name] = {
                    "position": smoothed_pos.tolist(),
                    "velocity": velocity.tolist(),
                    "visibility": landmark.visibility
                }
                
                # Update history
                self.joint_history[joint_name].append(smoothed_pos)
                if len(self.joint_history[joint_name]) > self.max_history:
                    self.joint_history[joint_name] = self.joint_history[joint_name][-self.max_history:]
        
        return joints
    
    def calculate_joint_angles(self, joints: Dict) -> Dict[str, float]:
        """Calculate key joint angles for biomechanical analysis."""
        angles = {}
        
        # Hip angle (torso-hip-knee)
        if all(j in joints for j in ["left_hip", "left_knee", "left_shoulder"]):
            angles["left_hip_angle"] = self._calculate_angle(
                joints["left_shoulder"]["position"],
                joints["left_hip"]["position"],
                joints["left_knee"]["position"]
            )
        
        if all(j in joints for j in ["right_hip", "right_knee", "right_shoulder"]):
            angles["right_hip_angle"] = self._calculate_angle(
                joints["right_shoulder"]["position"],
                joints["right_hip"]["position"],
                joints["right_knee"]["position"]
            )
        
        # Knee angle
        if all(j in joints for j in ["left_hip", "left_knee", "left_ankle"]):
            angles["left_knee_angle"] = self._calculate_angle(
                joints["left_hip"]["position"],
                joints["left_knee"]["position"],
                joints["left_ankle"]["position"]
            )
        
        if all(j in joints for j in ["right_hip", "right_knee", "right_ankle"]):
            angles["right_knee_angle"] = self._calculate_angle(
                joints["right_hip"]["position"],
                joints["right_knee"]["position"],
                joints["right_ankle"]["position"]
            )
        
        # Torso angle (vertical alignment)
        if all(j in joints for j in ["nose", "left_hip", "right_hip"]):
            hip_center = np.mean([
                joints["left_hip"]["position"],
                joints["right_hip"]["position"]
            ], axis=0)
            angles["torso_angle"] = self._calculate_vertical_angle(
                joints["nose"]["position"],
                hip_center
            )
        
        return angles
    
    @staticmethod
    def _calculate_angle(p1: List[float], p2: List[float], p3: List[float]) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    @staticmethod
    def _calculate_vertical_angle(p1: List[float], p2: List[float]) -> float:
        """Calculate angle from vertical (0° = upright, 90° = horizontal)."""
        p1 = np.array(p1)
        p2 = np.array(p2)
        
        vector = p2 - p1
        vertical = np.array([0, 1])  # Downward vertical
        
        cos_angle = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def get_joint_velocities(self, joints: Dict) -> Dict[str, float]:
        """Extract velocity magnitudes for all joints."""
        velocities = {}
        
        for joint_name, joint_data in joints.items():
            vx, vy = joint_data["velocity"]
            velocity_magnitude = np.sqrt(vx**2 + vy**2)
            velocities[joint_name] = velocity_magnitude
        
        return velocities
    
    def get_joint_accelerations(self, joint_name: str) -> float:
        """Calculate acceleration from velocity history."""
        if joint_name not in self.joint_history or len(self.joint_history[joint_name]) < 3:
            return 0.0
        
        # Get recent positions
        recent = self.joint_history[joint_name][-3:]
        
        # Calculate velocities
        v1 = recent[1] - recent[0]
        v2 = recent[2] - recent[1]
        
        # Acceleration
        acceleration = v2 - v1
        acceleration_magnitude = np.linalg.norm(acceleration)
        
        return acceleration_magnitude
    
    def get_center_of_mass(self, joints: Dict) -> Optional[Tuple[float, float]]:
        """Estimate center of mass from hip positions."""
        if "left_hip" in joints and "right_hip" in joints:
            left_hip = np.array(joints["left_hip"]["position"])
            right_hip = np.array(joints["right_hip"]["position"])
            com = (left_hip + right_hip) / 2.0
            return tuple(com)
        return None
    
    def reset(self):
        """Reset pose estimator state."""
        self._initialize_filters()
        for joint in self.joint_history:
            self.joint_history[joint] = []
        logger.info("Pose estimator reset")
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


def annotate_pose(frame: np.ndarray, joints: Optional[Dict], angles: Optional[Dict] = None) -> np.ndarray:
    """Annotate frame with pose skeleton and joint information."""
    if joints is None:
        return frame
    
    from utils.visualization import VideoAnnotator
    
    annotator = VideoAnnotator(frame.shape[1], frame.shape[0])
    
    # Prepare landmarks for drawing - ensure all positions are tuples with 2 elements
    landmarks = {}
    for name, data in joints.items():
        try:
            pos = data.get("position", None)
            if pos is not None:
                # Convert to array and flatten to ensure shape (2,)
                pos_array = np.asarray(pos).flatten()
                if pos_array.size >= 2:
                    landmarks[name] = tuple(pos_array[:2].astype(int))
        except (TypeError, ValueError, KeyError):
            # Skip malformed joint data
            continue
    
    # Draw skeleton only if we have valid landmarks
    if landmarks:
        frame = annotator.draw_skeleton(frame, landmarks)
    
    # Add joint angles as text
    if angles:
        y_offset = 150
        for angle_name, angle_value in angles.items():
            try:
                angle_val = float(angle_value)
                text = f"{angle_name}: {angle_val:.1f}°"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                y_offset += 20
            except (TypeError, ValueError):
                continue
    
    return frame


if __name__ == "__main__":
    # Example usage
    logger.info("Pose Estimator module loaded")

"""
Stage 4: Fall & Dangerous Event Detection
Rule-based detection of falls, collisions, and risky biomechanical events.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from loguru import logger

from config.config import FALL_CONFIG


class FallDetector:
    """Detect falls and dangerous events using biomechanical rules."""
    
    def __init__(self, frame_height: int):
        """Initialize fall detector with frame dimensions."""
        self.frame_height = frame_height
        
        # Event tracking
        self.fall_events: List[Dict] = []
        self.current_fall_state = False
        self.ground_contact_counter = 0
        
        # History for temporal analysis
        self.hip_position_history = deque(maxlen=30)
        self.torso_angle_history = deque(maxlen=30)
        self.head_position_history = deque(maxlen=30)
        self.velocity_history = deque(maxlen=10)
        
        logger.info("Fall Detector initialized")
    
    def detect_fall(
        self, 
        joints: Optional[Dict], 
        frame_num: int,
        com: Optional[Tuple[float, float]] = None
    ) -> Tuple[bool, Dict]:
        """
        Detect fall events using multiple biomechanical indicators.
        
        Returns: (is_falling, fall_info)
        """
        logger.debug(f"[STAGE 4 - FALL DETECTOR] Processing fall detection at frame {frame_num}")
        if joints is None:
            logger.debug(f"[STAGE 4 - FALL DETECTOR] No joints available, skipping fall detection")
            return False, {}
        
        fall_indicators = {
            "hip_drop": False,
            "torso_tilt": False,
            "head_velocity": False,
            "ground_contact": False,
            "sudden_stop": False
        }
        
        fall_severity = 0.0
        
        # Extract key positions
        hip_y = self._get_hip_height(joints)
        head_y = joints.get("nose", {}).get("position", [0, 0])[1]
        torso_angle = self._calculate_torso_angle(joints)
        
        # Update histories
        if hip_y is not None:
            self.hip_position_history.append(hip_y)
        if head_y is not None:
            self.head_position_history.append(head_y)
        if torso_angle is not None:
            self.torso_angle_history.append(torso_angle)
        
        # Indicator 1: Sudden hip drop
        if len(self.hip_position_history) >= 5:
            recent_hips = list(self.hip_position_history)[-5:]
            hip_drop_speed = (recent_hips[-1] - recent_hips[0]) / len(recent_hips)
            hip_drop_distance = recent_hips[-1] - min(recent_hips)
            
            if hip_drop_speed > FALL_CONFIG["hip_drop_speed_threshold"]:
                fall_indicators["hip_drop"] = True
                fall_severity += 0.3
            
            if hip_drop_distance > FALL_CONFIG["hip_drop_distance"]:
                fall_severity += 0.2
        
        # Indicator 2: Sudden torso orientation change
        if len(self.torso_angle_history) >= 5:
            recent_angles = list(self.torso_angle_history)[-5:]
            angle_change = abs(recent_angles[-1] - recent_angles[0])
            
            if angle_change > FALL_CONFIG["torso_angle_change_threshold"]:
                fall_indicators["torso_tilt"] = True
                fall_severity += 0.25
        
        # Indicator 3: Head downward velocity
        if len(self.head_position_history) >= 3:
            recent_heads = list(self.head_position_history)[-3:]
            head_velocity = (recent_heads[-1] - recent_heads[0]) / len(recent_heads)
            
            if head_velocity > FALL_CONFIG["head_downward_velocity"]:
                fall_indicators["head_velocity"] = True
                fall_severity += 0.2
        
        # Indicator 4: Ground contact (low vertical position)
        if hip_y is not None:
            normalized_height = hip_y / self.frame_height
            
            if normalized_height > FALL_CONFIG["ground_contact_height"]:
                self.ground_contact_counter += 1
                
                if self.ground_contact_counter >= FALL_CONFIG["ground_contact_duration"]:
                    fall_indicators["ground_contact"] = True
                    fall_severity += 0.15
            else:
                self.ground_contact_counter = 0
        
        # Indicator 5: Sudden stop after fast motion
        if com is not None:
            self.velocity_history.append(com)
            
            if len(self.velocity_history) >= 5:
                recent_positions = list(self.velocity_history)[-5:]
                
                # Calculate velocities
                velocities = []
                for i in range(1, len(recent_positions)):
                    dx = recent_positions[i][0] - recent_positions[i-1][0]
                    dy = recent_positions[i][1] - recent_positions[i-1][1]
                    vel = np.sqrt(dx**2 + dy**2)
                    velocities.append(vel)
                
                if len(velocities) >= 2:
                    deceleration = velocities[0] - velocities[-1]
                    
                    if deceleration > FALL_CONFIG["sudden_stop_deceleration"]:
                        fall_indicators["sudden_stop"] = True
                        fall_severity += 0.1
        
        # Determine if fall occurred
        conditions_met = sum(fall_indicators.values())
        is_falling = conditions_met >= FALL_CONFIG["fall_confirmation_conditions"]
        
        # Normalize severity
        fall_severity = min(fall_severity * 100, 100.0)
        
        # Record fall event
        if is_falling and not self.current_fall_state:
            fall_event = {
                "frame": frame_num,
                "severity": fall_severity,
                "indicators": fall_indicators.copy(),
                "hip_position": hip_y,
                "torso_angle": torso_angle
            }
            self.fall_events.append(fall_event)
            logger.warning(f"Fall detected at frame {frame_num} (severity: {fall_severity:.1f})")
        
        self.current_fall_state = is_falling
        
        fall_info = {
            "is_falling": is_falling,
            "severity": fall_severity,
            "indicators": fall_indicators,
            "conditions_met": conditions_met
        }
        
        return is_falling, fall_info
    
    def _get_hip_height(self, joints: Dict) -> Optional[float]:
        """Get average hip y-coordinate."""
        hip_positions = []
        
        for hip_name in ["left_hip", "right_hip"]:
            if hip_name in joints:
                hip_y = joints[hip_name]["position"][1]
                hip_positions.append(hip_y)
        
        if hip_positions:
            return np.mean(hip_positions)
        return None
    
    def _calculate_torso_angle(self, joints: Dict) -> Optional[float]:
        """Calculate torso angle from vertical."""
        if not all(j in joints for j in ["nose", "left_hip", "right_hip"]):
            return None
        
        # Calculate hip center
        left_hip = np.array(joints["left_hip"]["position"])
        right_hip = np.array(joints["right_hip"]["position"])
        hip_center = (left_hip + right_hip) / 2.0
        
        # Vector from hip to head
        nose = np.array(joints["nose"]["position"])
        torso_vector = nose - hip_center
        
        # Angle from vertical
        vertical = np.array([0, -1])  # Upward
        cos_angle = np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector) * np.linalg.norm(vertical) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def get_fall_summary(self) -> Dict:
        """Get summary of all detected falls."""
        if not self.fall_events:
            return {
                "total_falls": 0,
                "avg_severity": 0.0,
                "max_severity": 0.0,
                "fall_frames": []
            }
        
        return {
            "total_falls": len(self.fall_events),
            "avg_severity": np.mean([f["severity"] for f in self.fall_events]),
            "max_severity": max([f["severity"] for f in self.fall_events]),
            "fall_frames": [f["frame"] for f in self.fall_events],
            "events": self.fall_events
        }
    
    def reset(self):
        """Reset fall detector state."""
        self.fall_events.clear()
        self.current_fall_state = False
        self.ground_contact_counter = 0
        self.hip_position_history.clear()
        self.torso_angle_history.clear()
        self.head_position_history.clear()
        self.velocity_history.clear()
        logger.info("Fall detector reset")


def annotate_fall_detection(
    frame: np.ndarray, 
    fall_info: Dict,
    frame_num: int,
    alert_duration: int = 30
) -> np.ndarray:
    """Annotate frame with fall detection results."""
    import cv2
    from utils.visualization import VideoAnnotator
    
    annotator = VideoAnnotator(frame.shape[1], frame.shape[0])
    
    if fall_info.get("is_falling", False):
        # Draw alert
        frame = annotator.draw_event_alert(frame, "FALL", duration=alert_duration)
        
        # Show severity
        severity_text = f"Severity: {fall_info['severity']:.1f}%"
        cv2.putText(frame, severity_text, (frame.shape[1] - 250, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show indicators
    y_offset = 80
    for indicator, active in fall_info.get("indicators", {}).items():
        color = (0, 255, 0) if active else (128, 128, 128)
        text = f"{indicator}: {'YES' if active else 'NO'}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y_offset += 20
    
    return frame


if __name__ == "__main__":
    logger.info("Fall Detector module loaded")

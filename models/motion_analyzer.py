"""
Stage 5: Motion & Joint Stress Analysis
Temporal analysis of raider motion using sliding windows and biomechanical metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from loguru import logger

from config.config import MOTION_CONFIG


class MotionAnalyzer:
    """Analyze raider motion patterns and joint stress over time."""
    
    def __init__(self):
        """Initialize motion analyzer."""
        self.window_size = MOTION_CONFIG["window_size"]
        self.overlap = MOTION_CONFIG["overlap"]
        
        # Temporal buffers for each joint
        self.joint_position_buffer: Dict[str, deque] = {}
        self.joint_velocity_buffer: Dict[str, deque] = {}
        self.joint_acceleration_buffer: Dict[str, deque] = {}
        
        # Motion abnormality scores over time
        self.abnormality_scores = []
        
        # LSTM temporal analyzer (optional)
        self.lstm_analyzer = None
        try:
            from models.temporal_lstm import TemporalMotionAnalyzer
            # Try to load pretrained model if available
            from pathlib import Path
            model_path = Path(__file__).parent / "motion_lstm_model.pth"
            self.lstm_analyzer = TemporalMotionAnalyzer(
                model_path=str(model_path) if model_path.exists() else None
            )
            logger.info("LSTM temporal analyzer initialized")
        except Exception as e:
            logger.info(f"LSTM analyzer not available, using rule-based only: {e}")
            self.lstm_analyzer = None
        
        logger.info(f"Motion Analyzer initialized (window: {self.window_size} frames)")
    
    def update_joint_data(self, joints: Optional[Dict]):
        """Update temporal buffers with new joint data."""
        if joints is None:
            logger.debug(f"[STAGE 5 - MOTION ANALYZER] No joints available for update")
            return
        
        logger.debug(f"[STAGE 5 - MOTION ANALYZER] Updating joint data for {len(joints)} joints")
        
        for joint_name, joint_data in joints.items():
            # Initialize buffers if needed
            if joint_name not in self.joint_position_buffer:
                self.joint_position_buffer[joint_name] = deque(maxlen=self.window_size)
                self.joint_velocity_buffer[joint_name] = deque(maxlen=self.window_size)
                self.joint_acceleration_buffer[joint_name] = deque(maxlen=self.window_size)
                logger.debug(f"[STAGE 5 - MOTION ANALYZER] Initialized buffers for joint: {joint_name}")
            
            # Store position
            position = np.array(joint_data["position"])
            self.joint_position_buffer[joint_name].append(position)
            
            # Calculate and store velocity
            if len(self.joint_position_buffer[joint_name]) >= 2:
                prev_pos = self.joint_position_buffer[joint_name][-2]
                velocity = position - prev_pos
                self.joint_velocity_buffer[joint_name].append(velocity)
                
                # Calculate and store acceleration
                if len(self.joint_velocity_buffer[joint_name]) >= 2:
                    prev_vel = self.joint_velocity_buffer[joint_name][-2]
                    acceleration = velocity - prev_vel
                    self.joint_acceleration_buffer[joint_name].append(acceleration)
    
    def calculate_motion_metrics(self, joints: Optional[Dict]) -> Dict:
        """Calculate comprehensive motion metrics."""
        if joints is None:
            logger.debug(f"[STAGE 5 - MOTION ANALYZER] No joints available for motion metrics")
            return {}
        
        logger.debug(f"[STAGE 5 - MOTION ANALYZER] Calculating motion metrics")
        
        metrics = {
            "velocity": {},
            "acceleration": {},
            "jerk": {},
            "smoothness": {},
            "asymmetry": {},
        }
        
        # Update buffers
        self.update_joint_data(joints)
        
        # Calculate metrics for each joint
        for joint_name in joints.keys():
            if joint_name not in self.joint_position_buffer:
                continue
            
            logger.debug(f"[STAGE 5 - MOTION ANALYZER] Metrics for {joint_name}")
            
            # Velocity magnitude
            if len(self.joint_velocity_buffer[joint_name]) > 0:
                velocities = [np.linalg.norm(v) for v in self.joint_velocity_buffer[joint_name]]
                metrics["velocity"][joint_name] = {
                    "current": velocities[-1] if velocities else 0.0,
                    "max": max(velocities) if velocities else 0.0,
                    "avg": np.mean(velocities) if velocities else 0.0,
                    "std": np.std(velocities) if velocities else 0.0
                }
            
            # Acceleration magnitude
            if len(self.joint_acceleration_buffer[joint_name]) > 0:
                accelerations = [np.linalg.norm(a) for a in self.joint_acceleration_buffer[joint_name]]
                metrics["acceleration"][joint_name] = {
                    "current": accelerations[-1] if accelerations else 0.0,
                    "max": max(accelerations) if accelerations else 0.0,
                    "avg": np.mean(accelerations) if accelerations else 0.0
                }
            
            # Jerk (rate of acceleration change)
            if len(self.joint_acceleration_buffer[joint_name]) >= 2:
                accelerations = list(self.joint_acceleration_buffer[joint_name])
                jerks = []
                for i in range(1, len(accelerations)):
                    jerk = np.linalg.norm(accelerations[i] - accelerations[i-1])
                    jerks.append(jerk)
                
                if jerks:
                    metrics["jerk"][joint_name] = {
                        "current": jerks[-1],
                        "max": max(jerks),
                        "avg": np.mean(jerks)
                    }
            
            # Trajectory smoothness (lower is smoother)
            if len(self.joint_position_buffer[joint_name]) >= 3:
                positions = list(self.joint_position_buffer[joint_name])
                smoothness = self._calculate_smoothness(positions)
                metrics["smoothness"][joint_name] = smoothness
        
        # Calculate left-right asymmetry
        asymmetry = self._calculate_asymmetry(metrics)
        metrics["asymmetry"] = asymmetry
        
        return metrics
    
    def _calculate_smoothness(self, positions: List[np.ndarray]) -> float:
        """
        Calculate trajectory smoothness using jerk-based metric.
        Lower values = smoother motion
        """
        if len(positions) < 3:
            return 0.0
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(positions)):
            vel = positions[i] - positions[i-1]
            velocities.append(vel)
        
        # Calculate accelerations
        accelerations = []
        for i in range(1, len(velocities)):
            acc = velocities[i] - velocities[i-1]
            accelerations.append(acc)
        
        # Calculate jerks
        jerks = []
        for i in range(1, len(accelerations)):
            jerk = accelerations[i] - accelerations[i-1]
            jerks.append(np.linalg.norm(jerk))
        
        # Smoothness is inverse of average jerk
        if jerks:
            return np.mean(jerks)
        return 0.0
    
    def _calculate_asymmetry(self, metrics: Dict) -> Dict:
        """Calculate left-right asymmetry for paired joints."""
        asymmetry = {}
        
        paired_joints = [
            ("left_hip", "right_hip"),
            ("left_knee", "right_knee"),
            ("left_ankle", "right_ankle"),
            ("left_shoulder", "right_shoulder")
        ]
        
        for left_joint, right_joint in paired_joints:
            # Velocity asymmetry
            if left_joint in metrics["velocity"] and right_joint in metrics["velocity"]:
                left_vel = metrics["velocity"][left_joint]["avg"]
                right_vel = metrics["velocity"][right_joint]["avg"]
                
                if left_vel + right_vel > 0:
                    asym = abs(left_vel - right_vel) / (left_vel + right_vel)
                    asymmetry[f"{left_joint.split('_')[1]}_velocity"] = asym
            
            # Acceleration asymmetry
            if left_joint in metrics["acceleration"] and right_joint in metrics["acceleration"]:
                left_acc = metrics["acceleration"][left_joint]["avg"]
                right_acc = metrics["acceleration"][right_joint]["avg"]
                
                if left_acc + right_acc > 0:
                    asym = abs(left_acc - right_acc) / (left_acc + right_acc)
                    asymmetry[f"{left_joint.split('_')[1]}_acceleration"] = asym
        
        return asymmetry
    
    def calculate_abnormality_score(self, metrics: Dict) -> float:
        """
        Calculate overall motion abnormality score (0-100).
        Higher score = more abnormal/risky motion.
        Combines rule-based metrics with LSTM predictions when available.
        """
        if not metrics:
            return 0.0
        
        weights = MOTION_CONFIG["abnormality_weights"]
        rule_based_score = 0.0
        
        # Velocity score
        velocity_scores = []
        for joint_name, vel_data in metrics.get("velocity", {}).items():
            if vel_data["max"] > MOTION_CONFIG["velocity_threshold"]:
                velocity_scores.append(min(vel_data["max"] / MOTION_CONFIG["velocity_threshold"], 1.0))
        
        if velocity_scores:
            rule_based_score += weights["velocity"] * np.mean(velocity_scores) * 100
        
        # Acceleration score
        acceleration_scores = []
        for joint_name, acc_data in metrics.get("acceleration", {}).items():
            if acc_data["max"] > MOTION_CONFIG["acceleration_threshold"]:
                acceleration_scores.append(min(acc_data["max"] / MOTION_CONFIG["acceleration_threshold"], 1.0))
        
        if acceleration_scores:
            rule_based_score += weights["acceleration"] * np.mean(acceleration_scores) * 100
        
        # Jerk score
        jerk_scores = []
        for joint_name, jerk_data in metrics.get("jerk", {}).items():
            if jerk_data["max"] > MOTION_CONFIG["jerk_threshold"]:
                jerk_scores.append(min(jerk_data["max"] / MOTION_CONFIG["jerk_threshold"], 1.0))
        
        if jerk_scores:
            rule_based_score += weights["jerk"] * np.mean(jerk_scores) * 100
        
        # Asymmetry score
        asymmetry_values = list(metrics.get("asymmetry", {}).values())
        if asymmetry_values:
            avg_asymmetry = np.mean(asymmetry_values)
            if avg_asymmetry > MOTION_CONFIG["asymmetry_threshold"]:
                rule_based_score += weights["asymmetry"] * min(avg_asymmetry / MOTION_CONFIG["asymmetry_threshold"], 1.0) * 100
        
        # Smoothness deviation (higher jerk = less smooth = more abnormal)
        smoothness_values = list(metrics.get("smoothness", {}).values())
        if smoothness_values:
            avg_smoothness = np.mean(smoothness_values)
            # Normalize to 0-1 range (assuming typical range is 0-5)
            normalized_smoothness = min(avg_smoothness / 5.0, 1.0)
            rule_based_score += weights["deviation"] * normalized_smoothness * 100
        
        # Combine with LSTM prediction if available
        final_score = rule_based_score
        if self.lstm_analyzer and self.lstm_analyzer.use_lstm:
            # Get joints from buffer (reconstruct from position buffer)
            joints = self._reconstruct_joints_from_buffer()
            if joints:
                lstm_score = self.lstm_analyzer.predict_abnormality(joints)
                # Weighted combination: 70% rule-based, 30% LSTM
                final_score = 0.7 * rule_based_score + 0.3 * lstm_score
                logger.debug(f"Combined score: rule={rule_based_score:.1f}, lstm={lstm_score:.1f}, final={final_score:.1f}")
        
        # Store score
        self.abnormality_scores.append(min(final_score, 100.0))
        
        return min(final_score, 100.0)
    
    def _reconstruct_joints_from_buffer(self) -> Optional[Dict]:
        """Reconstruct joint dictionary from position buffer for LSTM."""
        if not self.joint_position_buffer:
            return None
        
        joints = {}
        for joint_name, positions in self.joint_position_buffer.items():
            if len(positions) > 0:
                joints[joint_name] = {
                    "position": list(positions[-1])
                }
        
        return joints if joints else None
    
    def get_temporal_trend(self, window: int = 30) -> Dict:
        """Get trend of abnormality scores over recent frames."""
        if len(self.abnormality_scores) < 2:
            return {"trend": "stable", "change": 0.0}
        
        recent = self.abnormality_scores[-window:]
        
        if len(recent) < 2:
            return {"trend": "stable", "change": 0.0}
        
        # Calculate trend
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)
        
        change = avg_second - avg_first
        
        if change > 5.0:
            trend = "increasing"
        elif change < -5.0:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change": change,
            "current_avg": avg_second,
            "previous_avg": avg_first
        }
    
    def get_summary_statistics(self) -> Dict:
        """Get summary of motion analysis."""
        if not self.abnormality_scores:
            return {
                "avg_abnormality": 0.0,
                "max_abnormality": 0.0,
                "min_abnormality": 0.0,
                "std_abnormality": 0.0
            }
        
        return {
            "avg_abnormality": np.mean(self.abnormality_scores),
            "max_abnormality": max(self.abnormality_scores),
            "min_abnormality": min(self.abnormality_scores),
            "std_abnormality": np.std(self.abnormality_scores),
            "total_frames": len(self.abnormality_scores)
        }
    
    def reset(self):
        """Reset motion analyzer state."""
        self.joint_position_buffer.clear()
        self.joint_velocity_buffer.clear()
        self.joint_acceleration_buffer.clear()
        self.abnormality_scores.clear()
        logger.info("Motion analyzer reset")


def annotate_motion_analysis(frame: np.ndarray, metrics: Dict, abnormality_score: float) -> np.ndarray:
    """Annotate frame with motion analysis results."""
    import cv2
    from utils.visualization import VideoAnnotator
    
    annotator = VideoAnnotator(frame.shape[1], frame.shape[0])
    
    # Display abnormality score
    try:
        score = float(abnormality_score)
        score_text = f"Motion Abnormality: {score:.1f}%"
        color = (0, 255, 0) if score < 30 else (0, 165, 255) if score < 70 else (0, 0, 255)
        cv2.putText(frame, score_text, (10, frame.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    except (TypeError, ValueError):
        pass
    
    # Display key metrics
    stats = {}
    
    try:
        if "velocity" in metrics and metrics["velocity"] and isinstance(metrics["velocity"], dict):
            velocities_list = []
            for v in metrics["velocity"].values():
                if isinstance(v, dict) and "avg" in v:
                    try:
                        velocities_list.append(float(v["avg"]))
                    except (TypeError, ValueError):
                        pass
            if velocities_list:
                stats["Avg Velocity"] = f"{np.mean(velocities_list):.1f} px/f"
    except (TypeError, AttributeError):
        pass
    
    try:
        if "acceleration" in metrics and metrics["acceleration"] and isinstance(metrics["acceleration"], dict):
            accelerations_list = []
            for a in metrics["acceleration"].values():
                if isinstance(a, dict) and "max" in a:
                    try:
                        accelerations_list.append(float(a["max"]))
                    except (TypeError, ValueError):
                        pass
            if accelerations_list:
                stats["Max Accel"] = f"{max(accelerations_list):.1f} px/f²"
    except (TypeError, AttributeError):
        pass
    
    try:
        if "asymmetry" in metrics and metrics["asymmetry"] and isinstance(metrics["asymmetry"], dict):
            asymmetry_values = []
            for val in metrics["asymmetry"].values():
                try:
                    asymmetry_values.append(float(val))
                except (TypeError, ValueError):
                    pass
            if asymmetry_values:
                stats["Asymmetry"] = f"{np.mean(asymmetry_values):.2f}"
    except (TypeError, AttributeError):
        pass
    
    if stats:
        frame = annotator.draw_stats_panel(frame, stats, position=(frame.shape[1] - 360, 10))
    
    return frame


if __name__ == "__main__":
    logger.info("Motion Analyzer module loaded")

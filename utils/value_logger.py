"""
Simple Value Logger for Intermediate Calculation Capture
Non-intrusive logging that happens AFTER frame processing completes.
Does NOT affect pipeline logic - just stores raw values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger


class ValueLogger:
    """Log calculation values after each frame is processed."""
    
    def __init__(self, output_dir: Path, max_frames: int = 1000):
        """
        Initialize value logger.
        
        Args:
            output_dir: Session output directory
            max_frames: Maximum frames to log (default 1000)
        """
        self.output_dir = Path(output_dir)
        self.max_frames = max_frames
        self.frame_count = 0
        self.enabled = True
        
        # Create output subdirectory
        self.log_dir = self.output_dir / "calculation_values"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"[VALUE_LOGGER] Initialized: {self.log_dir}")
        
        # Simple lists to store raw values (NO processing)
        self.yolo_log = []
        self.tracking_log = []
        self.raider_log = []
        self.pose_log = []
        self.fall_log = []
        self.motion_log = []
        self.impact_log = []
        self.risk_log = []
    
    def can_log(self) -> bool:
        """Check if we should continue logging."""
        return self.frame_count < self.max_frames and self.enabled
    
    def log_frame(self, frame_num: int, data: Dict) -> None:
        """
        Log all values for a completed frame.
        Called AFTER all 7 stages complete.
        
        Args:
            frame_num: Frame number
            data: Dictionary with all stage outputs
        """
        if not self.can_log():
            return
        
        try:
            # Log Stage 1: YOLO Detections and Tracking
            # Extract player data (which contains tracking info)
            if "players" in data and data["players"] is not None:
                for idx, player in enumerate(data["players"]):
                    self.yolo_log.append({
                        "frame_num": frame_num,
                        "detection_id": idx,
                        "track_id": player.get("track_id"),
                        "x1": player.get("bbox", [0, 0, 0, 0])[0],
                        "y1": player.get("bbox", [0, 0, 0, 0])[1],
                        "x2": player.get("bbox", [0, 0, 0, 0])[2],
                        "y2": player.get("bbox", [0, 0, 0, 0])[3]
                    })
                    
                    # Also log tracking info
                    self.tracking_log.append({
                        "frame_num": frame_num,
                        "track_id": int(player.get("track_id", 0)),
                        "x1": player.get("bbox", [0, 0, 0, 0])[0],
                        "y1": player.get("bbox", [0, 0, 0, 0])[1],
                        "x2": player.get("bbox", [0, 0, 0, 0])[2],
                        "y2": player.get("bbox", [0, 0, 0, 0])[3]
                    })
            
            # Log Stage 2: Raider
            if "raider_info" in data and data["raider_info"] is not None:
                raider_info = data["raider_info"]
                self.raider_log.append({
                    "frame_num": frame_num,
                    "raider_id": raider_info.get("track_id"),
                    "confidence": raider_info.get("confidence", 0.0),
                    "speed": raider_info.get("speed", 0.0),
                    "is_raider": True
                })
            elif "players" in data and data["players"]:
                # Log all players as non-raider
                for player in data["players"]:
                    self.raider_log.append({
                        "frame_num": frame_num,
                        "raider_id": player.get("track_id"),
                        "confidence": 0.0,
                        "speed": 0.0,
                        "is_raider": False
                    })
            
            # Log Stage 3: Pose
            if "joints" in data and data["joints"] is not None:
                joints = data["joints"]
                pose_record = {
                    "frame_num": frame_num,
                    "has_pose": True,
                    "joints_detected": len(joints) if isinstance(joints, dict) else 0
                }
                
                # Extract joint coordinates, velocities, and distances
                if isinstance(joints, dict) and len(joints) > 0:
                    # Store joint positions and velocities
                    for joint_name, joint_data in joints.items():
                        if isinstance(joint_data, dict):
                            pos = joint_data.get("position", [0, 0])
                            vel = joint_data.get("velocity", [0, 0])
                            
                            pose_record[f"{joint_name}_x"] = pos[0] if len(pos) > 0 else 0
                            pose_record[f"{joint_name}_y"] = pos[1] if len(pos) > 1 else 0
                            pose_record[f"{joint_name}_vx"] = vel[0] if len(vel) > 0 else 0
                            pose_record[f"{joint_name}_vy"] = vel[1] if len(vel) > 1 else 0
                    
                    # Calculate key euclidean distances
                    try:
                        # Shoulder distance
                        if "left_shoulder" in joints and "right_shoulder" in joints:
                            l_sho = joints["left_shoulder"].get("position", [0, 0])
                            r_sho = joints["right_shoulder"].get("position", [0, 0])
                            pose_record["shoulder_distance"] = np.sqrt(
                                (l_sho[0] - r_sho[0])**2 + (l_sho[1] - r_sho[1])**2
                            )
                        
                        # Hip distance
                        if "left_hip" in joints and "right_hip" in joints:
                            l_hip = joints["left_hip"].get("position", [0, 0])
                            r_hip = joints["right_hip"].get("position", [0, 0])
                            pose_record["hip_distance"] = np.sqrt(
                                (l_hip[0] - r_hip[0])**2 + (l_hip[1] - r_hip[1])**2
                            )
                        
                        # Left arm length (shoulder to wrist)
                        if "left_shoulder" in joints and "left_wrist" in joints:
                            l_sho = joints["left_shoulder"].get("position", [0, 0])
                            l_wri = joints["left_wrist"].get("position", [0, 0])
                            pose_record["left_arm_length"] = np.sqrt(
                                (l_sho[0] - l_wri[0])**2 + (l_sho[1] - l_wri[1])**2
                            )
                        
                        # Right arm length (shoulder to wrist)
                        if "right_shoulder" in joints and "right_wrist" in joints:
                            r_sho = joints["right_shoulder"].get("position", [0, 0])
                            r_wri = joints["right_wrist"].get("position", [0, 0])
                            pose_record["right_arm_length"] = np.sqrt(
                                (r_sho[0] - r_wri[0])**2 + (r_sho[1] - r_wri[1])**2
                            )
                        
                        # Left leg length (hip to ankle)
                        if "left_hip" in joints and "left_ankle" in joints:
                            l_hip = joints["left_hip"].get("position", [0, 0])
                            l_ank = joints["left_ankle"].get("position", [0, 0])
                            pose_record["left_leg_length"] = np.sqrt(
                                (l_hip[0] - l_ank[0])**2 + (l_hip[1] - l_ank[1])**2
                            )
                        
                        # Right leg length (hip to ankle)
                        if "right_hip" in joints and "right_ankle" in joints:
                            r_hip = joints["right_hip"].get("position", [0, 0])
                            r_ank = joints["right_ankle"].get("position", [0, 0])
                            pose_record["right_leg_length"] = np.sqrt(
                                (r_hip[0] - r_ank[0])**2 + (r_hip[1] - r_ank[1])**2
                            )
                        
                        # Torso length (nose to hip center)
                        if "nose" in joints and "left_hip" in joints and "right_hip" in joints:
                            nose = joints["nose"].get("position", [0, 0])
                            l_hip = joints["left_hip"].get("position", [0, 0])
                            r_hip = joints["right_hip"].get("position", [0, 0])
                            hip_center_x = (l_hip[0] + r_hip[0]) / 2
                            hip_center_y = (l_hip[1] + r_hip[1]) / 2
                            pose_record["torso_length"] = np.sqrt(
                                (nose[0] - hip_center_x)**2 + (nose[1] - hip_center_y)**2
                            )
                        
                        # Joint velocity magnitude (overall movement)
                        joint_speeds = []
                        for joint_name, joint_data in joints.items():
                            if isinstance(joint_data, dict):
                                vel = joint_data.get("velocity", [0, 0])
                                speed = np.sqrt(vel[0]**2 + vel[1]**2)
                                joint_speeds.append(speed)
                        
                        if joint_speeds:
                            pose_record["avg_joint_velocity"] = np.mean(joint_speeds)
                            pose_record["max_joint_velocity"] = np.max(joint_speeds)
                        
                    except Exception as e:
                        logger.warning(f"Error calculating pose distances: {e}")
                
                self.pose_log.append(pose_record)
            
            # Log Stage 4: Fall
            if "fall_info" in data and data["fall_info"] is not None:
                fall_info = data["fall_info"]
                self.fall_log.append({
                    "frame_num": frame_num,
                    "is_falling": data.get("is_falling", False),
                    "fall_severity": fall_info.get("fall_severity", 0.0)
                })
            
            # Log Stage 5: Motion
            if "motion_abnormality" in data:
                self.motion_log.append({
                    "frame_num": frame_num,
                    "abnormality_score": data.get("motion_abnormality", 0.0)
                })
            
            # Log Stage 6: Impact
            if "impact_detected" in data:
                self.impact_log.append({
                    "frame_num": frame_num,
                    "impact_detected": data.get("impact_detected", False),
                    "impact_severity": data.get("impact_severity", 0.0)
                })
            
            # Log Stage 7: Risk
            if "risk_data" in data and data["risk_data"] is not None:
                risk_data = data["risk_data"]
                self.risk_log.append({
                    "frame_num": frame_num,
                    "risk_score": risk_data.get("risk_score", 0.0),
                    "risk_level": risk_data.get("risk_level", "low")
                })
            
            self.frame_count = frame_num
        
        except Exception as e:
            logger.warning(f"[VALUE_LOGGER] Error logging frame {frame_num}: {e}")
    
    def export(self) -> bool:
        """
        Export all logged values to CSV files.
        Called at the end of pipeline run.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"[VALUE_LOGGER] Exporting {self.frame_count} frames to {self.log_dir}")
            
            if self.yolo_log:
                pd.DataFrame(self.yolo_log).to_csv(
                    self.log_dir / "YOLO_detection_values.csv", 
                    index=False
                )
                logger.debug(f"[VALUE_LOGGER] Saved {len(self.yolo_log)} YOLO records")
            
            if self.tracking_log:
                pd.DataFrame(self.tracking_log).to_csv(
                    self.log_dir / "SORT_tracking_values.csv", 
                    index=False
                )
                logger.debug(f"[VALUE_LOGGER] Saved {len(self.tracking_log)} tracking records")
            
            if self.raider_log:
                pd.DataFrame(self.raider_log).to_csv(
                    self.log_dir / "Raider_identification_values.csv", 
                    index=False
                )
                logger.debug(f"[VALUE_LOGGER] Saved {len(self.raider_log)} raider records")
            
            if self.pose_log:
                pd.DataFrame(self.pose_log).to_csv(
                    self.log_dir / "Pose_estimation_values.csv", 
                    index=False
                )
                logger.debug(f"[VALUE_LOGGER] Saved {len(self.pose_log)} pose records")
            
            if self.fall_log:
                pd.DataFrame(self.fall_log).to_csv(
                    self.log_dir / "Fall_detection_values.csv", 
                    index=False
                )
                logger.debug(f"[VALUE_LOGGER] Saved {len(self.fall_log)} fall records")
            
            if self.motion_log:
                pd.DataFrame(self.motion_log).to_csv(
                    self.log_dir / "Motion_analysis_values.csv", 
                    index=False
                )
                logger.debug(f"[VALUE_LOGGER] Saved {len(self.motion_log)} motion records")
            
            if self.impact_log:
                pd.DataFrame(self.impact_log).to_csv(
                    self.log_dir / "Impact_detection_values.csv", 
                    index=False
                )
                logger.debug(f"[VALUE_LOGGER] Saved {len(self.impact_log)} impact records")
            
            if self.risk_log:
                pd.DataFrame(self.risk_log).to_csv(
                    self.log_dir / "Risk_fusion_values.csv", 
                    index=False
                )
                logger.debug(f"[VALUE_LOGGER] Saved {len(self.risk_log)} risk records")
            
            logger.info(f"[VALUE_LOGGER] Export complete to {self.log_dir}")
            return True
        
        except Exception as e:
            logger.warning(f"[VALUE_LOGGER] Export failed: {e}")
            return False

"""
Stage 6: Defender Interaction & Impact Modeling
Detect defender attacks and estimate collision intensity without full pose estimation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from loguru import logger

from config.config import IMPACT_CONFIG


class ImpactDetector:
    """Detect and quantify defender-raider collisions and impacts."""
    
    def __init__(self):
        """Initialize impact detector."""
        self.collision_radius = IMPACT_CONFIG["collision_radius"]
        self.approach_radius = IMPACT_CONFIG["approach_radius"]
        
        # Track defender positions over time
        self.defender_history: Dict[int, deque] = {}
        self.max_history = 30
        
        # Impact events
        self.impact_events: List[Dict] = []
        
        logger.info(f"Impact Detector initialized (collision radius: {self.collision_radius}px)")
    
    def update_defender_positions(self, players: List[Dict], raider_id: int):
        """Update position history for defenders (non-raider players)."""
        logger.debug(f"[STAGE 6 - IMPACT DETECTOR] Updating defender positions (raider_id: {raider_id})")
        current_defender_ids = set()
        
        for player in players:
            track_id = player["track_id"]
            
            # Skip raider
            if track_id == raider_id:
                continue
            
            current_defender_ids.add(track_id)
            
            # Initialize history
            if track_id not in self.defender_history:
                self.defender_history[track_id] = deque(maxlen=self.max_history)
                logger.debug(f"[STAGE 6 - IMPACT DETECTOR] New defender tracked - ID: {track_id}")
            
            # Store position and bbox
            self.defender_history[track_id].append({
                "center": player["center"],
                "bbox": player["bbox"],
                "frame": player.get("frame_num", 0)
            })
        
        # Clean up old defenders
        old_ids = set(self.defender_history.keys()) - current_defender_ids
        for old_id in old_ids:
            if len(self.defender_history[old_id]) == 0:
                del self.defender_history[old_id]
                logger.debug(f"[STAGE 6 - IMPACT DETECTOR] Removed defender ID: {old_id}")
    
    def calculate_defender_velocity(self, defender_id: int, num_frames: int = 5) -> np.ndarray:
        """Calculate defender velocity vector."""
        if defender_id not in self.defender_history:
            return np.array([0.0, 0.0])
        
        history = list(self.defender_history[defender_id])
        if len(history) < 2:
            return np.array([0.0, 0.0])
        
        recent = history[-min(num_frames, len(history)):]
        
        # Calculate average velocity
        velocities = []
        for i in range(1, len(recent)):
            p1 = np.array(recent[i-1]["center"])
            p2 = np.array(recent[i]["center"])
            vel = p2 - p1
            velocities.append(vel)
        
        if velocities:
            return np.mean(velocities, axis=0)
        return np.array([0.0, 0.0])
    
    def calculate_approach_angle(
        self, 
        defender_pos: Tuple[float, float],
        defender_velocity: np.ndarray,
        raider_pos: Tuple[float, float]
    ) -> float:
        """
        Calculate angle between defender velocity and direction to raider.
        0° = directly toward raider, 90° = perpendicular
        """
        defender_pos = np.array(defender_pos)
        raider_pos = np.array(raider_pos)
        
        # Vector from defender to raider
        to_raider = raider_pos - defender_pos
        
        # Normalize vectors
        vel_norm = np.linalg.norm(defender_velocity)
        to_raider_norm = np.linalg.norm(to_raider)
        
        if vel_norm < 1e-6 or to_raider_norm < 1e-6:
            return 90.0  # No approach
        
        defender_velocity = defender_velocity / vel_norm
        to_raider = to_raider / to_raider_norm
        
        # Calculate angle
        cos_angle = np.dot(defender_velocity, to_raider)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def detect_impacts(
        self, 
        players: List[Dict], 
        raider_id: Optional[int],
        frame_num: int,
        collision_impact_score: float = 0.0
    ) -> Tuple[bool, Dict]:
        """
        Detect defender impacts on raider.
        
        Args:
            players: List of all players
            raider_id: Track ID of raider
            frame_num: Current frame number
            collision_impact_score: Impact score from collision detection (0.0-1.0)
        
        Returns: (impact_detected, impact_info)
        """
        logger.debug(f"[STAGE 6 - IMPACT DETECTOR] Detecting impacts at frame {frame_num}, raider_id: {raider_id}")
        if raider_id is None:
            logger.debug(f"[STAGE 6 - IMPACT DETECTOR] No raider ID provided, skipping impact detection")
            return False, {}
        
        # Find raider
        raider = next((p for p in players if p["track_id"] == raider_id), None)
        if raider is None:
            logger.debug(f"[STAGE 6 - IMPACT DETECTOR] Raider not found in player list")
            return False, {}
        
        raider_pos = np.array(raider["center"])
        logger.debug(f"[STAGE 6 - IMPACT DETECTOR] Raider position: {raider_pos}")
        
        # Use collision impact score if available
        if collision_impact_score > 0.0:
            logger.debug(f"[STAGE 6 - IMPACT DETECTOR] Using collision impact score: {collision_impact_score:.2f}")
        
        # Update defender positions
        self.update_defender_positions(players, raider_id)
        
        # Analyze each defender
        approaching_defenders = []
        colliding_defenders = []
        
        for defender_id, history in self.defender_history.items():
            if len(history) == 0:
                continue
            
            defender_data = history[-1]
            defender_pos = np.array(defender_data["center"])
            
            # Calculate distance to raider
            distance = np.linalg.norm(raider_pos - defender_pos)
            
            # Calculate defender velocity
            velocity = self.calculate_defender_velocity(defender_id)
            velocity_magnitude = np.linalg.norm(velocity)
            
            # Check if approaching
            if distance < self.approach_radius and velocity_magnitude > IMPACT_CONFIG["min_approach_velocity"]:
                approach_angle = self.calculate_approach_angle(defender_pos, velocity, raider_pos)
                
                if approach_angle < IMPACT_CONFIG["approach_angle_tolerance"]:
                    approaching_defenders.append({
                        "id": defender_id,
                        "distance": distance,
                        "velocity": velocity_magnitude,
                        "angle": approach_angle
                    })
                    logger.debug(f"[STAGE 6 - IMPACT DETECTOR] Defender {defender_id} approaching at distance {distance:.2f}")
            
            # Check if colliding
            if distance < self.collision_radius:
                if velocity_magnitude > IMPACT_CONFIG["collision_velocity_threshold"]:
                    colliding_defenders.append({
                        "id": defender_id,
                        "distance": distance,
                        "velocity": velocity_magnitude
                    })
                    logger.debug(f"[STAGE 6 - IMPACT DETECTOR] COLLISION with defender {defender_id}")
        
        # Calculate impact severity
        impact_detected = len(colliding_defenders) > 0 or len(approaching_defenders) >= IMPACT_CONFIG["convergence_threshold"]
        impact_severity = self._calculate_impact_severity(
            approaching_defenders,
            colliding_defenders,
            raider_pos
        )
        
        # Incorporate collision impact score
        if collision_impact_score > 0.0:
            # Blend collision score with calculated severity
            # Collision score is 0-1.0, convert to 0-100 scale
            collision_severity = collision_impact_score * 100
            # Weighted average: 60% motion-based, 40% collision-based
            impact_severity = (0.6 * impact_severity) + (0.4 * collision_severity)
            impact_detected = True
            logger.debug(f"[STAGE 6 - IMPACT DETECTOR] Blended impact with collision score: {impact_severity:.1f}")
        
        logger.debug(f"[STAGE 6 - IMPACT DETECTOR] Impact detected: {impact_detected}, severity: {impact_severity:.3f}")
        
        # Record impact event
        if impact_detected:
            impact_event = {
                "frame": frame_num,
                "severity": impact_severity,
                "num_colliding": len(colliding_defenders),
                "num_approaching": len(approaching_defenders),
                "colliding_defenders": colliding_defenders,
                "approaching_defenders": approaching_defenders,
                "collision_score": collision_impact_score
            }
            self.impact_events.append(impact_event)
            logger.debug(f"Impact detected at frame {frame_num} (severity: {impact_severity:.1f})")
        
        impact_info = {
            "impact_detected": impact_detected,
            "severity": impact_severity,
            "num_colliding": len(colliding_defenders),
            "num_approaching": len(approaching_defenders),
            "approaching_defenders": approaching_defenders,
            "colliding_defenders": colliding_defenders,
            "collision_score": collision_impact_score
        }
        
        return impact_detected, impact_info
    
    def _calculate_impact_severity(
        self,
        approaching_defenders: List[Dict],
        colliding_defenders: List[Dict],
        raider_pos: np.ndarray
    ) -> float:
        """Calculate impact severity score (0-100)."""
        severity = 0.0
        weights = IMPACT_CONFIG["impact_weights"]
        
        # Velocity component
        if colliding_defenders:
            avg_collision_velocity = np.mean([d["velocity"] for d in colliding_defenders])
            velocity_score = min(avg_collision_velocity / IMPACT_CONFIG["collision_velocity_threshold"], 1.0)
            severity += weights["velocity"] * velocity_score * 100
        
        if approaching_defenders:
            avg_approach_velocity = np.mean([d["velocity"] for d in approaching_defenders])
            velocity_score = min(avg_approach_velocity / IMPACT_CONFIG["min_approach_velocity"], 1.0)
            severity += weights["velocity"] * velocity_score * 50  # Less weight for approaching
        
        # Distance component (closer = higher severity)
        if colliding_defenders:
            avg_distance = np.mean([d["distance"] for d in colliding_defenders])
            distance_score = 1.0 - (avg_distance / self.collision_radius)
            severity += weights["distance"] * distance_score * 100
        
        # Angle component (direct approach = higher severity)
        if approaching_defenders:
            avg_angle = np.mean([d["angle"] for d in approaching_defenders])
            angle_score = 1.0 - (avg_angle / IMPACT_CONFIG["approach_angle_tolerance"])
            severity += weights["angle"] * angle_score * 100
        
        # Number of defenders
        total_defenders = len(colliding_defenders) + len(approaching_defenders)
        num_defenders_score = min(total_defenders / IMPACT_CONFIG["max_defenders_considered"], 1.0)
        severity += weights["num_defenders"] * num_defenders_score * 100
        
        # Convergence (multiple defenders from different directions)
        if total_defenders >= IMPACT_CONFIG["convergence_threshold"]:
            severity += weights["convergence"] * 100
        
        return min(severity, 100.0)
    
    def get_impact_summary(self) -> Dict:
        """Get summary of all detected impacts."""
        if not self.impact_events:
            return {
                "total_impacts": 0,
                "avg_severity": 0.0,
                "max_severity": 0.0,
                "impact_frames": []
            }
        
        return {
            "total_impacts": len(self.impact_events),
            "avg_severity": np.mean([e["severity"] for e in self.impact_events]),
            "max_severity": max([e["severity"] for e in self.impact_events]),
            "impact_frames": [e["frame"] for e in self.impact_events],
            "events": self.impact_events
        }
    
    def reset(self):
        """Reset impact detector state."""
        self.defender_history.clear()
        self.impact_events.clear()
        logger.info("Impact detector reset")


def annotate_impact_detection(
    frame: np.ndarray,
    players: List[Dict],
    raider_id: Optional[int],
    impact_info: Dict
) -> np.ndarray:
    """Annotate frame with impact detection results."""
    import cv2
    from utils.visualization import VideoAnnotator
    
    annotator = VideoAnnotator(frame.shape[1], frame.shape[0])
    
    if not impact_info.get("impact_detected", False):
        return frame
    
    # Find raider position
    raider = next((p for p in players if p["track_id"] == raider_id), None)
    if raider is None:
        return frame
    
    raider_center = raider["center"]
    
    # Draw collision radius
    cv2.circle(frame, raider_center, IMPACT_CONFIG["collision_radius"], (0, 165, 255), 2)
    
    # Highlight colliding defenders
    for defender_info in impact_info.get("colliding_defenders", []):
        defender_id = defender_info["id"]
        defender = next((p for p in players if p["track_id"] == defender_id), None)
        if defender:
            frame = annotator.draw_bbox(frame, defender["bbox"], "COLLISION!", color=(0, 0, 255), thickness=3)
    
    # Show approaching defenders
    for defender_info in impact_info.get("approaching_defenders", []):
        defender_id = defender_info["id"]
        defender = next((p for p in players if p["track_id"] == defender_id), None)
        if defender:
            frame = annotator.draw_bbox(frame, defender["bbox"], "APPROACHING", color=(0, 165, 255))
            # Draw line from defender to raider
            cv2.line(frame, defender["center"], raider_center, (0, 165, 255), 2)
    
    # Show impact alert
    if impact_info["severity"] > 50:
        frame = annotator.draw_event_alert(frame, "IMPACT", duration=30)
    
    # Show severity
    severity_text = f"Impact Severity: {impact_info['severity']:.1f}%"
    cv2.putText(frame, severity_text, (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame


if __name__ == "__main__":
    logger.info("Impact Detector module loaded")

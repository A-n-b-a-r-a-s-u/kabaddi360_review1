"""
Stage 2: Raider Identification
Multi-cue raider identification using court logic, motion, and temporal consistency.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
from loguru import logger

from config.config import RAIDER_CONFIG


class RaiderIdentifier:
    """Identify the raider using multiple cues without relying solely on color."""
    
    def __init__(self, frame_width: int, frame_height: int):
        """Initialize raider identifier with court dimensions."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Court midline (vertical center)
        self.midline_y = frame_height * RAIDER_CONFIG["midline_position"]
        self.crossing_threshold = frame_height * RAIDER_CONFIG["crossing_threshold"]
        
        # Motion tracking - 90 frames (~3 seconds at 30fps) for initial evaluation window
        self.player_motion_history: Dict[int, deque] = {}
        self.max_history_length = 90  # Changed from 30 to 90 frames
        self.evaluation_window = 90  # Evaluation window for metric calculation
        
        # Raider state - permanent locking after first identification
        self.current_raider_id: Optional[int] = None
        self.raider_locked = False  # Flag indicating permanent lock
        
        # Confidence tracking
        self.raider_confidence: Dict[int, float] = {}
        self.metrics_evaluated = False  # Track if metrics have been evaluated
        
        logger.info(f"Raider Identifier initialized. Midline at y={self.midline_y:.1f}")
        logger.info(f"Evaluation window: {self.evaluation_window} frames (~3 seconds at 30fps)")
    
    def update_player_positions(self, players: List[Dict]):
        """Update position history for all players."""
        current_ids = set()
        
        for player in players:
            track_id = player["track_id"]
            center = player["center"]
            
            current_ids.add(track_id)
            
            # Initialize history if new player
            if track_id not in self.player_motion_history:
                self.player_motion_history[track_id] = deque(maxlen=self.max_history_length)
            
            # Add current position
            self.player_motion_history[track_id].append({
                "position": center,
                "bbox": player["bbox"],
                "timestamp": player.get("frame_num", 0)
            })
        
        # Clean up old tracks
        old_ids = set(self.player_motion_history.keys()) - current_ids
        for old_id in old_ids:
            if len(self.player_motion_history[old_id]) == 0:
                del self.player_motion_history[old_id]
    
    def calculate_speed(self, track_id: int) -> float:
        """
        Calculate average speed over the entire evaluation window (90 frames).
        Computed once using cumulative displacement.
        
        Returns: average speed in pixels per frame over the 90-frame window
        """
        if track_id not in self.player_motion_history:
            return 0.0
        
        history = list(self.player_motion_history[track_id])
        
        # Need full evaluation window
        if len(history) < self.evaluation_window:
            return 0.0
        
        # Use only the first 90 frames for consistent evaluation
        eval_history = history[:self.evaluation_window]
        
        if len(eval_history) < 2:
            return 0.0
        
        # Calculate total displacement
        total_distance = 0.0
        for i in range(1, len(eval_history)):
            x1, y1 = eval_history[i-1]["position"]
            x2, y2 = eval_history[i]["position"]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance
        
        # Average speed over window
        avg_speed = total_distance / len(eval_history)
        
        logger.debug(f"[STAGE 2 - RAIDER] Track {track_id}: Avg speed over {self.evaluation_window} frames = {avg_speed:.2f} px/frame")
        
        return avg_speed
    
    def calculate_direction_changes(self, track_id: int) -> int:
        """
        Count direction changes over the evaluation window (90 frames).
        Evaluated once during initial identification.
        
        Returns: number of direction changes where heading differs by ≥45°
        """
        if track_id not in self.player_motion_history:
            return 0
        
        history = list(self.player_motion_history[track_id])
        
        # Need full evaluation window
        if len(history) < self.evaluation_window:
            return 0
        
        # Use only the first 90 frames
        eval_history = history[:self.evaluation_window]
        
        if len(eval_history) < 3:
            return 0
        
        changes = 0
        for i in range(2, len(eval_history)):
            # Calculate movement vectors
            dx1 = eval_history[i-1]["position"][0] - eval_history[i-2]["position"][0]
            dy1 = eval_history[i-1]["position"][1] - eval_history[i-2]["position"][1]
            dx2 = eval_history[i]["position"][0] - eval_history[i-1]["position"][0]
            dy2 = eval_history[i]["position"][1] - eval_history[i-1]["position"][1]
            
            # Calculate angle change
            if dx1 != 0 or dy1 != 0:
                angle1 = np.arctan2(dy1, dx1)
                angle2 = np.arctan2(dy2, dx2)
                angle_diff = abs(np.degrees(angle2 - angle1))
                
                # Normalize to 0-180
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                if angle_diff > RAIDER_CONFIG["direction_change_threshold"]:
                    changes += 1
        
        logger.debug(f"[STAGE 2 - RAIDER] Track {track_id}: Direction changes = {changes} over {self.evaluation_window} frames")
        
        return changes
    
    def calculate_directional_commitment(self, track_id: int) -> float:
        """
        Calculate directional commitment: consistency of movement direction over 90 frames.
        Evaluated once during initial identification.
        
        Raiders move with purpose, defenders react (changing direction).
        
        Returns: float between 0 and 1
        - 1.0 = moving in same direction consistently (≥10% of frames)
        - 0.0 = random/reactive movement
        """
        if track_id not in self.player_motion_history:
            return 0.0
        
        history = list(self.player_motion_history[track_id])
        
        # Need full evaluation window
        if len(history) < self.evaluation_window:
            return 0.0
        
        # Use only the first 90 frames
        eval_history = history[:self.evaluation_window]
        
        if len(eval_history) < 5:
            return 0.0
        
        # Calculate movement vectors
        vectors = []
        for i in range(1, len(eval_history)):
            x1, y1 = eval_history[i-1]["position"]
            x2, y2 = eval_history[i]["position"]
            
            dx = x2 - x1
            dy = y2 - y1
            magnitude = np.sqrt(dx**2 + dy**2)
            
            if magnitude > 0.5:  # Only consider significant movements
                angle = np.arctan2(dy, dx)
                vectors.append(angle)
        
        if len(vectors) < 2:
            return 0.0
        
        # Check how many consecutive frames move in similar direction
        consistent_frames = 0
        total_movements = len(vectors)
        
        for i in range(len(vectors) - 1):
            angle_diff = abs(np.degrees(vectors[i+1] - vectors[i]))
            
            # Normalize to 0-180
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # If angle diff < 45°, consider it same direction
            if angle_diff < 45:
                consistent_frames += 1
        
        # Raider commits to direction at least 10% of frames
        commitment_ratio = consistent_frames / max(total_movements, 1)
        
        logger.debug(f"[STAGE 2 - RAIDER] Track {track_id}: Directional commitment = {commitment_ratio:.2f} over {self.evaluation_window} frames")
        
        return min(commitment_ratio, 1.0)
    
    def check_midline_crossing(self, track_id: int) -> bool:
        """Check if player has crossed the midline (court logic)."""
        if track_id not in self.player_motion_history:
            return False
        
        history = list(self.player_motion_history[track_id])
        if len(history) < 2:
            return False
        
        # Check recent positions
        recent = history[-5:]
        
        # Determine if player crossed from one half to another
        positions_above = sum(1 for p in recent if p["position"][1] < self.midline_y - self.crossing_threshold)
        positions_below = sum(1 for p in recent if p["position"][1] > self.midline_y + self.crossing_threshold)
        
        # Crossing detected if player moved from one side to other
        return positions_above > 0 and positions_below > 0
    
    def calculate_raider_confidence(self, track_id: int, players: List[Dict]) -> float:
        """
        Calculate confidence that this player is the raider.
        Uses 4-cue system evaluated once over 90-frame window:
        1. Directional Commitment (30%) - Consistent movement direction
        2. Speed Dominance (30%) - Among fastest players
        3. Direction Changes (20%) - Dodging behavior
        4. Role Persistence (20%) - NOT USED (permanent lock after first identification)
        
        This evaluation is performed only once when raider_id is None.
        """
        if track_id not in self.player_motion_history:
            logger.debug(f"[STAGE 2 - RAIDER] Track ID {track_id} not in motion history")
            return 0.0
        
        confidence = 0.0
        
        # CUE 1: Directional Commitment (30% weight)
        # Raiders move with purpose, not randomly
        directional_score = self.calculate_directional_commitment(track_id)
        if directional_score >= 0.1:  # At least 10% of frames moving consistently
            cue1 = min(directional_score / 0.5, 1.0) * 0.3  # Scale up to max 0.3
            confidence += cue1
            logger.debug(f"[STAGE 2 - RAIDER] Track {track_id}: Directional commitment = {directional_score:.3f} → +{cue1:.3f}")
        
        # CUE 2: Speed Dominance (30% weight)
        # Raiders are usually among the fastest players
        speed = self.calculate_speed(track_id)
        all_speeds = [self.calculate_speed(p["track_id"]) for p in players]
        if all_speeds:
            speed_percentile = sum(1 for s in all_speeds if speed > s) / len(all_speeds)
            if speed_percentile >= RAIDER_CONFIG["speed_percentile"] / 100.0:
                confidence += 0.3
                logger.debug(f"[STAGE 2 - RAIDER] Track {track_id}: Speed dominance detected (percentile: {speed_percentile:.2f}) → +0.30")
        
        # CUE 3: Direction Changes (20% weight)
        # Raiders dodge defenders (change direction frequently)
        direction_changes = self.calculate_direction_changes(track_id)
        if direction_changes >= 2:
            confidence += 0.2
            logger.debug(f"[STAGE 2 - RAIDER] Track {track_id}: Direction changes detected ({direction_changes}) → +0.20")
        
        # CUE 4: Role Persistence (20%) - NOT USED
        # With permanent locking, role persistence is not needed
        # Each player starts fresh evaluation from scratch
        
        logger.info(f"[STAGE 2 - RAIDER] Track {track_id}: Final confidence = {confidence:.3f}")
        
        return confidence
    
    
    def identify_raider(self, players: List[Dict]) -> Optional[int]:
        """
        Identify the raider from players list using 4-cue system.
        
        Evaluation Logic:
        - First call (raider_id is None): Calculate metrics over 90-frame window
        - Once identified with confidence ≥70%: LOCK permanently for rest of clip
        - Subsequent calls: Return locked raider_id without re-evaluation
        
        Returns: track_id of raider, or None if not identified yet
        """
        if not players:
            return None
        
        # If raider already locked, return it without re-evaluation
        if self.raider_locked and self.current_raider_id is not None:
            logger.debug(f"[STAGE 2 - RAIDER] Raider locked: Track {self.current_raider_id}")
            return self.current_raider_id
        
        # First evaluation - perform once when sufficient history is available
        if not self.metrics_evaluated:
            self.update_player_positions(players)
            
            # Wait for evaluation window to fill (90 frames)
            min_history = min(len(self.player_motion_history[p["track_id"]]) for p in players if p["track_id"] in self.player_motion_history)
            
            if min_history < self.evaluation_window:
                logger.debug(f"[STAGE 2 - RAIDER] Waiting for evaluation window ({min_history}/{self.evaluation_window} frames)")
                return None
            
            # Perform metric evaluation
            logger.info(f"[STAGE 2 - RAIDER] Evaluation window reached ({self.evaluation_window} frames). Identifying raider...")
            
            player_scores = {}
            for player in players:
                track_id = player["track_id"]
                confidence = self.calculate_raider_confidence(track_id, players)
                player_scores[track_id] = confidence
            
            # Find best candidate
            if player_scores:
                best_raider_id = max(player_scores.items(), key=lambda x: x[1])[0]
                best_confidence = player_scores[best_raider_id]
                
                if best_confidence >= RAIDER_CONFIG["confidence_threshold"]:
                    # Lock raider permanently
                    self.current_raider_id = best_raider_id
                    self.raider_locked = True
                    self.raider_confidence[best_raider_id] = best_confidence
                    self.metrics_evaluated = True
                    
                    logger.info(f"[STAGE 2 - RAIDER] ✓ RAIDER IDENTIFIED & LOCKED: Track {best_raider_id} (confidence: {best_confidence:.3f})")
                    logger.info(f"[STAGE 2 - RAIDER] Raider remains locked for entire clip duration")
                    
                    return best_raider_id
                else:
                    self.metrics_evaluated = True
                    logger.warning(f"[STAGE 2 - RAIDER] No confident raider found. Best candidate: Track {best_raider_id} (confidence: {best_confidence:.3f})")
                    return None
        
        return None
    
    def get_raider_info(self, players: List[Dict]) -> Optional[Dict]:
        """Get full information about the identified raider."""
        raider_id = self.identify_raider(players)
        
        if raider_id is None:
            return None
        
        # Find raider in players list
        raider_data = next((p for p in players if p["track_id"] == raider_id), None)
        
        if raider_data:
            raider_data["confidence"] = self.raider_confidence.get(raider_id, 0.0)
            raider_data["speed"] = self.calculate_speed(raider_id)
            raider_data["is_raider"] = True
        
        return raider_data
    
    def reset(self):
        """Reset raider identification state."""
        self.current_raider_id = None
        self.raider_locked = False
        self.metrics_evaluated = False
        self.raider_confidence.clear()
        self.player_motion_history.clear()
        logger.info("Raider identifier reset")


def annotate_raider(frame: np.ndarray, players: List[Dict], raider_info: Optional[Dict]) -> np.ndarray:
    """Annotate frame with raider identification."""
    from utils.visualization import VideoAnnotator
    
    annotator = VideoAnnotator(frame.shape[1], frame.shape[0])
    
    # Draw all players
    for player in players:
        is_raider = raider_info and player["track_id"] == raider_info["track_id"]
        
        if is_raider:
            # Highlight raider
            frame = annotator.draw_raider_highlight(frame, player["bbox"])
        else:
            # Regular player box
            frame = annotator.draw_bbox(
                frame, 
                player["bbox"], 
                f"ID: {player['track_id']}", 
                color=(255, 0, 0)  # Blue for defenders
            )
    
    # Add raider info
    if raider_info:
        stats = {
            "Raider ID": raider_info["track_id"],
            "Confidence": f"{raider_info['confidence']:.2f}",
            "Speed": f"{raider_info['speed']:.1f} px/frame"
        }
        frame = annotator.draw_stats_panel(frame, stats, position=(10, 10))
    else:
        stats = {"Raider": "NOT DETECTED"}
        frame = annotator.draw_stats_panel(frame, stats, position=(10, 10))
    
    return frame


if __name__ == "__main__":
    # Example usage for testing
    logger.info("Raider Identifier module loaded")

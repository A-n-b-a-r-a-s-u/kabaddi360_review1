"""
Raider Status Tracker: Tracks raider state and generates events for EventLogger.
Monitors raider detection, state changes, and interactions.
"""

from typing import Optional, Dict, List
from loguru import logger


class RaiderStatusTracker:
    """Track raider state and generate appropriate events."""
    
    STATES = {
        "UNKNOWN": "Unknown",
        "WAIT": "⏳ Waiting for detection",
        "MOVING": "Moving",
        "FALLEN": "Fallen",
        "TOUCHED": "Touched",
        "EXITED": "Exited Court"
    }
    
    def __init__(self):
        """Initialize raider status tracker."""
        self.raider_id: Optional[int] = None
        self.raider_confidence: float = 0.0
        self.current_state: str = self.STATES["WAIT"]
        self.previous_state: str = self.STATES["WAIT"]
        self.detection_time: Optional[float] = None
        self.touch_count: int = 0
        self.last_touched_defender_id: Optional[int] = None
        self.fall_count: int = 0
        
        # Frame tracking
        self.frames_since_detection: int = 0
        self.last_event_frame: int = 0
        
        logger.info("RaiderStatusTracker initialized")
    
    def set_raider_detected(
        self,
        raider_id: int,
        confidence: float,
        detection_time: float,
        event_logger = None
    ):
        """
        Set raider as detected.
        
        Args:
            raider_id: Track ID of detected raider
            confidence: Detection confidence (0-100)
            detection_time: Timestamp in seconds
            event_logger: EventLogger instance to log the event
        """
        if self.raider_id != raider_id:
            self.raider_id = raider_id
            self.raider_confidence = confidence
            self.detection_time = detection_time
            self.touch_count = 0
            self.fall_count = 0
            self.current_state = self.STATES["MOVING"]
            
            if event_logger:
                event_logger.log_event(
                    "RAIDER_DETECTED",
                    detection_time,
                    {
                        "raider_id": raider_id,
                        "confidence": confidence,
                        "timestamp": detection_time
                    }
                )
            
            logger.info(f"[RAIDER STATUS] Raider detected: ID={raider_id}, Confidence={confidence:.1f}%")
    
    def raider_entered_court(self, frame_num: int, timestamp: float, event_logger = None):
        """Log raider entering court."""
        if self.raider_id and frame_num != self.last_event_frame:
            if event_logger:
                event_logger.log_event(
                    "RAIDER_ENTERED",
                    timestamp,
                    {
                        "raider_id": self.raider_id,
                        "frame": frame_num
                    }
                )
            self.last_event_frame = frame_num
            logger.info(f"[RAIDER STATUS] Raider {self.raider_id} entered court at {timestamp:.2f}s")
    
    def set_raider_state(
        self,
        new_state: str,
        frame_num: int,
        timestamp: float,
        details: Dict = None,
        event_logger = None
    ):
        """
        Update raider state.
        
        Args:
            new_state: New state (MOVING, FALLEN, TOUCHED, etc.)
            frame_num: Current frame number
            timestamp: Current timestamp in seconds
            details: State-specific details
            event_logger: EventLogger instance
        """
        if details is None:
            details = {}
        
        state_display = self.STATES.get(new_state, new_state)
        
        if self.current_state != state_display:
            self.previous_state = self.current_state
            self.current_state = state_display
            
            if event_logger:
                event_logger.log_event(
                    "STATE_CHANGE",
                    timestamp,
                    {
                        "raider_id": self.raider_id,
                        "old_state": self.previous_state,
                        "new_state": state_display,
                        "frame": frame_num,
                        **details
                    }
                )
            
            logger.debug(f"[RAIDER STATUS] State changed: {self.previous_state} → {state_display}")
    
    def fall_detected(
        self,
        frame_num: int,
        timestamp: float,
        severity: str,
        event_logger = None
    ):
        """
        Log fall detection.
        
        Args:
            frame_num: Frame number
            timestamp: Timestamp in seconds
            severity: Fall severity (Low, Medium, High)
            event_logger: EventLogger instance
        """
        if self.raider_id:
            self.fall_count += 1
            self.current_state = self.STATES["FALLEN"]
            
            if event_logger:
                event_logger.log_event(
                    "FALL_DETECTED",
                    timestamp,
                    {
                        "raider_id": self.raider_id,
                        "severity": severity,
                        "frame": frame_num,
                        "fall_count": self.fall_count
                    }
                )
            
            logger.info(f"[RAIDER STATUS] Fall detected: Severity={severity}, Count={self.fall_count}")
    
    def raider_touched(
        self,
        defender_id: int,
        frame_num: int,
        timestamp: float,
        event_logger = None
    ):
        """
        Log raider being touched by defender.
        
        Args:
            defender_id: ID of touching defender
            frame_num: Frame number
            timestamp: Timestamp in seconds
            event_logger: EventLogger instance
        """
        if self.raider_id:
            self.touch_count += 1
            self.last_touched_defender_id = defender_id
            
            if event_logger:
                event_logger.log_event(
                    "RAIDER_TOUCHED",
                    timestamp,
                    {
                        "raider_id": self.raider_id,
                        "defender_id": defender_id,
                        "frame": frame_num,
                        "touch_count": self.touch_count
                    }
                )
            
            logger.debug(f"[RAIDER STATUS] Raider {self.raider_id} touched by defender {defender_id} (touch #{self.touch_count})")
    
    def get_status(self) -> Dict:
        """Get current raider status dictionary."""
        return {
            "raider_id": self.raider_id,
            "confidence": self.raider_confidence,
            "state": self.current_state,
            "detected_at": self.detection_time,
            "touch_count": self.touch_count,
            "fall_count": self.fall_count,
            "last_touched_defender": self.last_touched_defender_id,
            "is_detected": self.raider_id is not None
        }
    
    def reset(self):
        """Reset for new video processing."""
        self.raider_id = None
        self.raider_confidence = 0.0
        self.current_state = self.STATES["WAIT"]
        self.previous_state = self.STATES["WAIT"]
        self.detection_time = None
        self.touch_count = 0
        self.last_touched_defender_id = None
        self.fall_count = 0
        self.frames_since_detection = 0
        self.last_event_frame = 0
        
        logger.info("[RAIDER STATUS] Tracker reset for new session")

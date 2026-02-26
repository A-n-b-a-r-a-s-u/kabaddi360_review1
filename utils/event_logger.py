"""
Event Logger: Thread-safe event queue with JSON persistence
Logs raider events during video processing for real-time dashboard display.
"""

import json
import threading
from queue import Queue
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger


class EventLogger:
    """Thread-safe event logger with queue-based persistence."""
    
    # Event types
    EVENT_TYPES = {
        "RAIDER_DETECTED": "🎯 Raider Detected",
        "RAIDER_ENTERED": "⚡ Raider Entered Court",
        "RAIDER_EXITED": "🚪 Raider Exited Court",
        "FALL_DETECTED": "❌ Fall Detected",
        "RAIDER_TOUCHED": "💥 Raider Touched Defender",
        "STATE_CHANGE": "🔄 State Changed",
        "FRAME_UPDATE": "📊 Status Update"
    }
    
    def __init__(self, output_dir: Path):
        """
        Initialize event logger.
        
        Args:
            output_dir: Session output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Events file paths
        self.live_events_file = self.output_dir / "live_events.json"
        self.final_events_file = self.output_dir / "events_timeline.json"
        
        # Thread-safe queue
        self.event_queue: Queue = Queue()
        self.lock = threading.RLock()
        
        # Event counter and buffer
        self.event_count = 0
        self.frame_count = 0
        self.events_buffer: List[Dict] = []
        
        # Status variables
        self.current_raider_id: Optional[int] = None
        self.current_raider_confidence: float = 0.0
        self.current_raider_state: str = "Unknown"  # Moving, Fallen, Touched
        self.raider_touch_count: int = 0
        self.raider_detected_time: Optional[float] = None
        
        logger.info(f"EventLogger initialized - Output: {self.output_dir}")
    
    def log_event(
        self,
        event_type: str,
        timestamp: float,  # seconds from start
        details: Dict[str, Any] = None
    ):
        """
        Log an event to the queue.
        
        Args:
            event_type: Type of event (RAIDER_DETECTED, FALL_DETECTED, etc.)
            timestamp: Seconds from video start
            details: Event-specific details
        """
        if details is None:
            details = {}
        
        event = {
            "type": event_type,
            "timestamp": timestamp,
            "time_str": self._format_timestamp(timestamp),
            "message": self.EVENT_TYPES.get(event_type, event_type),
            "details": details,
            "logged_at": datetime.now().isoformat()
        }
        
        with self.lock:
            self.event_queue.put(event)
            self.events_buffer.append(event)
            self.event_count += 1
            
            # Update status based on event
            self._update_status(event_type, details)
        
        logger.debug(f"[EVENT LOGGER] {event_type}: {event['message']}")
    
    def _update_status(self, event_type: str, details: Dict):
        """Update raider status based on event type."""
        if event_type == "RAIDER_DETECTED":
            self.current_raider_id = details.get("raider_id")
            self.current_raider_confidence = details.get("confidence", 0.0)
            self.raider_detected_time = details.get("timestamp")
            self.raider_touch_count = 0
            self.current_raider_state = "Moving"
        
        elif event_type == "FALL_DETECTED":
            self.current_raider_state = "Fallen"
        
        elif event_type == "RAIDER_TOUCHED":
            self.current_raider_state = "Touched"
            self.raider_touch_count += 1
        
        elif event_type == "STATE_CHANGE":
            self.current_raider_state = details.get("new_state", "Unknown")
    
    def update_frame(self, frame_num: int):
        """Update frame counter for flush trigger."""
        self.frame_count = frame_num
    
    def flush_events(self) -> bool:
        """
        Flush queued events to JSON file.
        Returns True if events were written, False if queue was empty.
        """
        if self.event_queue.empty():
            return False
        
        with self.lock:
            # Collect all queued events
            events_to_write = list(self.events_buffer)
            
            if not events_to_write:
                return False
            
            # Write to live events file (for dashboard polling)
            try:
                live_data = {
                    "last_update": datetime.now().isoformat(),
                    "total_events": self.event_count,
                    "current_frame": self.frame_count,
                    "events": events_to_write[-10:],  # Keep last 10 for UI
                    "raider_status": {
                        "id": self.current_raider_id,
                        "confidence": self.current_raider_confidence,
                        "state": self.current_raider_state,
                        "detected_at": self.raider_detected_time,
                        "touch_count": self.raider_touch_count
                    }
                }
                
                with open(self.live_events_file, 'w') as f:
                    json.dump(live_data, f, indent=2)
                
                logger.debug(f"[EVENT LOGGER] Flushed {len(events_to_write)} events to file")
                return True
            except Exception as e:
                logger.error(f"[EVENT LOGGER] Failed to flush events: {e}")
                return False
    
    def finalize_events(self):
        """Finalize events - write complete timeline to final file."""
        with self.lock:
            # Ensure all events are flushed
            self.flush_events()
            
            try:
                final_data = {
                    "session_summary": {
                        "total_events": self.event_count,
                        "raider_id": self.current_raider_id,
                        "raider_detected_at": self.raider_detected_time,
                        "total_touches": self.raider_touch_count,
                        "final_state": self.current_raider_state,
                        "finalized_at": datetime.now().isoformat()
                    },
                    "timeline": self.events_buffer
                }
                
                with open(self.final_events_file, 'w') as f:
                    json.dump(final_data, f, indent=2)
                
                logger.info(f"[EVENT LOGGER] Finalized {len(self.events_buffer)} events")
            except Exception as e:
                logger.error(f"[EVENT LOGGER] Failed to finalize events: {e}")
    
    def get_live_events(self) -> Optional[Dict]:
        """
        Get live events from file (for dashboard polling).
        
        Returns:
            Dictionary with live events or None if file doesn't exist
        """
        try:
            if self.live_events_file.exists():
                with open(self.live_events_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"[EVENT LOGGER] Failed to read live events: {e}")
        
        return None
    
    def get_raider_status(self) -> Dict:
        """Get current raider status."""
        with self.lock:
            return {
                "id": self.current_raider_id,
                "confidence": self.current_raider_confidence,
                "state": self.current_raider_state,
                "detected_at": self.raider_detected_time,
                "touch_count": self.raider_touch_count,
                "detected": self.current_raider_id is not None
            }
    
    def reset(self):
        """Reset event logger for new processing session."""
        with self.lock:
            while not self.event_queue.empty():
                try:
                    self.event_queue.get_nowait()
                except:
                    break
            
            self.events_buffer = []
            self.event_count = 0
            self.frame_count = 0
            self.current_raider_id = None
            self.current_raider_confidence = 0.0
            self.current_raider_state = "Unknown"
            self.raider_touch_count = 0
            self.raider_detected_time = None
            
            logger.info("[EVENT LOGGER] Reset for new session")
    
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format timestamp as MM:SS."""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes:02d}:{secs:02d}"

"""
Stage 1: Player Detection & Tracking
YOLO-based person detection with SORT tracking for consistent IDs.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics not installed. Run: pip install ultralytics")
    raise

from utils.tracking_utils import SORTTracker
from utils.visualization import VideoAnnotator
from config.config import YOLO_CONFIG, TRACKING_CONFIG


class PlayerDetector:
    """Detect and track all players in the frame using YOLO + SORT."""
    
    def __init__(self):
        """Initialize YOLO model and tracker."""
        logger.info("Initializing Player Detector...")
        
        # Load YOLO model
        try:
            self.model = YOLO(YOLO_CONFIG["model"])
            logger.info(f"YOLO model loaded: {YOLO_CONFIG['model']}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        # Initialize tracker
        self.tracker = SORTTracker(
            max_age=TRACKING_CONFIG["max_age"],
            min_hits=TRACKING_CONFIG["min_hits"],
            iou_threshold=TRACKING_CONFIG["iou_threshold"]
        )
        
        # Detection parameters
        self.confidence = YOLO_CONFIG["confidence"]
        self.iou_threshold = YOLO_CONFIG["iou_threshold"]
        self.device = YOLO_CONFIG["device"]
        
        # Track history
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        
        logger.info("Player Detector initialized successfully")
    
    def detect_players(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect all persons in the frame.
        Returns: array of detections [[x1, y1, x2, y2, confidence], ...]
        """
        logger.debug(f"[STAGE 1 - PLAYER DETECTOR] Detecting players in frame shape: {frame.shape}")
        # Run YOLO detection
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=[0],  # person class only
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            logger.debug(f"[STAGE 1 - PLAYER DETECTOR] Found {len(boxes)} players")
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())  # Ensure scalar value
                
                detections.append([float(x1), float(y1), float(x2), float(y2), conf])
        else:
            logger.debug(f"[STAGE 1 - PLAYER DETECTOR] No detections found in frame")
        
        return np.array(detections, dtype=np.float32) if detections else np.empty((0, 5), dtype=np.float32)
    
    def update_tracks(self, detections: np.ndarray) -> np.ndarray:
        """
        Update tracker with new detections.
        Returns: array of tracks [[x1, y1, x2, y2, track_id], ...]
        """
        logger.debug(f"[STAGE 1 - PLAYER DETECTOR] Updating tracks with {len(detections)} detections")
        tracks = self.tracker.update(detections)
        logger.debug(f"[STAGE 1 - PLAYER DETECTOR] Track update complete, {len(tracks)} active tracks")
        
        # Update track history for trajectory
        for track in tracks:
            track_id = int(track[4])
            cx = int((track[0] + track[2]) / 2)
            cy = int((track[1] + track[3]) / 2)
            
            if track_id not in self.track_history:
                self.track_history[track_id] = []
                logger.debug(f"[STAGE 1 - PLAYER DETECTOR] New track ID: {track_id}")
            
            self.track_history[track_id].append((cx, cy))
            
            # Keep only recent history (last 30 points)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id] = self.track_history[track_id][-30:]
        
        return tracks
    
    def process_frame(self, frame: np.ndarray, frame_num: int = 0) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process single frame: detect and track players.
        Returns: (annotated_frame, player_data)
        """
        # Detect players
        detections = self.detect_players(frame)
        
        # Update tracks
        tracks = self.update_tracks(detections)
        
        # Prepare player data
        players = []
        for track in tracks:
            player_data = {
                "track_id": int(track[4]),
                "bbox": track[:4].tolist(),
                "center": (int((track[0] + track[2]) / 2), int((track[1] + track[3]) / 2)),
                "frame_num": frame_num
            }
            players.append(player_data)
        
        # Annotate frame
        annotated_frame = self.annotate_frame(frame.copy(), tracks)
        
        return annotated_frame, players
    
    def annotate_frame(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """Draw bounding boxes and track IDs on frame."""
        annotator = VideoAnnotator(frame.shape[1], frame.shape[0])
        
        for track in tracks:
            track_id = int(track[4])
            bbox = track[:4]
            
            # Draw bounding box with track ID
            label = f"ID: {track_id}"
            frame = annotator.draw_bbox(frame, bbox, label, color=(0, 255, 0))
            
            # Draw trajectory if available
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                frame = annotator.draw_trajectory(
                    frame, 
                    self.track_history[track_id], 
                    color=(0, 255, 255),
                    thickness=2
                )
        
        # Add stats
        stats = {
            "Players Detected": len(tracks),
            "Active Tracks": len(self.track_history)
        }
        frame = annotator.draw_stats_panel(frame, stats, position=(10, 10))
        
        return frame
    
    def get_track_history(self, track_id: int, num_frames: int = 30) -> List[Tuple[int, int]]:
        """Get recent position history for a track."""
        if track_id in self.track_history:
            return self.track_history[track_id][-num_frames:]
        return []
    
    def reset(self):
        """Reset tracker and history."""
        self.tracker = SORTTracker(
            max_age=TRACKING_CONFIG["max_age"],
            min_hits=TRACKING_CONFIG["min_hits"],
            iou_threshold=TRACKING_CONFIG["iou_threshold"]
        )
        self.track_history.clear()
        logger.info("Player detector reset")


def run_player_detection(
    video_path: str,
    output_path: str,
    max_frames: Optional[int] = None,
    save_data: bool = True
) -> Dict:
    """
    Run player detection on entire video.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        max_frames: Maximum frames to process (None = all)
        save_data: Whether to save detection data to JSON
    
    Returns:
        Dictionary with detection results and statistics
    """
    from utils.video_utils import VideoReader, VideoWriter
    import json
    from tqdm import tqdm
    
    logger.info(f"Starting player detection on {video_path}")
    
    # Initialize detector
    detector = PlayerDetector()
    
    # Open video
    reader = VideoReader(video_path)
    
    # Prepare output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = VideoWriter(
        str(output_path),
        reader.fps,
        reader.width,
        reader.height
    )
    
    # Process frames
    all_detections = []
    total_frames = min(reader.total_frames, max_frames) if max_frames else reader.total_frames
    
    try:
        with tqdm(total=total_frames, desc="Detecting Players") as pbar:
            for frame_num, frame in reader.read_frames(max_frames):
                # Process frame
                annotated_frame, players = detector.process_frame(frame, frame_num)
                
                # Write output
                writer.write_frame(annotated_frame)
                
                # Save detection data
                all_detections.append({
                    "frame": frame_num,
                    "players": players
                })
                
                pbar.update(1)
    
    finally:
        reader.release()
        writer.release()
    
    # Save detection data
    if save_data:
        data_path = output_path.parent / f"{output_path.stem}_detections.json"
        with open(data_path, 'w') as f:
            json.dump(all_detections, f, indent=2)
        logger.info(f"Detection data saved to {data_path}")
    
    # Calculate statistics
    total_detections = sum(len(d["players"]) for d in all_detections)
    avg_detections = total_detections / max(len(all_detections), 1)
    unique_tracks = len(detector.track_history)
    
    results = {
        "output_video": str(output_path),
        "total_frames": len(all_detections),
        "total_detections": total_detections,
        "avg_detections_per_frame": avg_detections,
        "unique_tracks": unique_tracks,
        "detections": all_detections
    }
    
    logger.info(f"Player detection completed. Processed {len(all_detections)} frames, "
                f"{unique_tracks} unique tracks, avg {avg_detections:.1f} players/frame")
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python player_detector.py <input_video> <output_video>")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    
    results = run_player_detection(input_video, output_video)
    print(f"\nResults: {results}")

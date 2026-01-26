"""
Video utilities for reading, writing, and processing video files.
Handles frame extraction, encoding, and basic video operations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Generator
from loguru import logger


class VideoReader:
    """Read video frames with error handling and frame management."""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video loaded: {self.width}x{self.height} @ {self.fps}fps, {self.total_frames} frames")
    
    def read_frame(self) -> Optional[Tuple[bool, np.ndarray]]:
        """Read next frame from video."""
        ret, frame = self.cap.read()
        return ret, frame if ret else None
    
    def read_frames(self, max_frames: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator to iterate through video frames."""
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            yield frame_idx, frame
            frame_idx += 1
            
            if max_frames and frame_idx >= max_frames:
                break
    
    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """Get specific frame by number."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoWriter:
    """Write frames to video file with consistent encoding."""
    
    def __init__(self, output_path: str, fps: int, width: int, height: int, codec: str = "mp4v"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.width = width
        self.height = height
        
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # Initialize writer
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output_path}")
        
        logger.info(f"Video writer created: {output_path} ({width}x{height} @ {fps}fps)")
    
    def write_frame(self, frame: np.ndarray):
        """Write a single frame to video."""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)
    
    def release(self):
        """Release video writer."""
        if self.writer:
            self.writer.release()
            logger.info(f"Video saved: {self.output_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def resize_frame(frame: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    """Resize frame maintaining aspect ratio."""
    if width is None and height is None:
        return frame
    
    h, w = frame.shape[:2]
    
    if width and height:
        return cv2.resize(frame, (width, height))
    elif width:
        aspect_ratio = h / w
        new_height = int(width * aspect_ratio)
        return cv2.resize(frame, (width, new_height))
    else:  # height
        aspect_ratio = w / h
        new_width = int(height * aspect_ratio)
        return cv2.resize(frame, (new_width, height))


def extract_frames(video_path: str, output_dir: str, step: int = 1) -> int:
    """Extract frames from video to directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with VideoReader(video_path) as reader:
        for idx, frame in reader.read_frames():
            if idx % step == 0:
                frame_path = output_dir / f"frame_{idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                count += 1
    
    logger.info(f"Extracted {count} frames to {output_dir}")
    return count


def create_video_from_frames(frames_dir: str, output_path: str, fps: int = 30, pattern: str = "*.jpg"):
    """Create video from directory of frames."""
    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob(pattern))
    
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir} with pattern {pattern}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    height, width = first_frame.shape[:2]
    
    with VideoWriter(output_path, fps, width, height) as writer:
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            writer.write_frame(frame)
    
    logger.info(f"Created video from {len(frame_files)} frames")


def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    
    info = {
        "path": video_path,
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(1, int(cap.get(cv2.CAP_PROP_FPS))),
    }
    
    cap.release()
    return info

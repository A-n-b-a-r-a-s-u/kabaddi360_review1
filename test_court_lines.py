"""
Test script for Stage 0: Court Line Detection
"""

import cv2
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.court_line_detector import CourtLineDetector
from config.config import STAGE_DIRS

def test_court_line_detection(video_path: str, max_frames: int = 100):
    """Test court line detection on a video."""
    print(f"Testing court line detection on: {video_path}")
    
    # Initialize detector
    detector = CourtLineDetector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Create output directory
    output_dir = STAGE_DIRS[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, lines = detector.process_frame(frame, frame_count)
        
        # Save every 30th frame
        if frame_count % 30 == 0:
            output_path = output_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"Frame {frame_count}: Detected {len(lines)} lines - Saved to {output_path}")
        
        frame_count += 1
    
    cap.release()
    print(f"\nTest completed! Processed {frame_count} frames")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_court_lines.py <video_path> [max_frames]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    test_court_line_detection(video_path, max_frames)

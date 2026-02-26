"""
Debug script to test court line detection on a single frame.
Run this to verify court lines are being detected in your video.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from models.court_line_detector import CourtLineDetector
from config.config import STAGE_DIRS

def test_court_line_detection(video_path: str, frame_num: int = 0):
    """Test court line detection on a specific frame."""
    
    print(f"\n{'='*60}")
    print("COURT LINE DETECTION DEBUG TEST")
    print(f"{'='*60}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return False
    
    # Get specified frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Failed to read frame {frame_num}")
        return False
    
    print(f"✓ Video loaded: {frame.shape[1]}x{frame.shape[0]}")
    print(f"✓ Processing frame {frame_num}...")
    
    # Initialize detector
    detector = CourtLineDetector()
    
    # Process frame
    try:
        annotated_frame, line_mapping = detector.process_frame(frame, frame_num)
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return False
    
    print(f"\n{'='*60}")
    print("DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Lines detected: {len(line_mapping)}")
    
    if len(line_mapping) == 0:
        print("\n⚠️  NO LINES DETECTED!")
        print("\nPossible causes:")
        print("1. HSV range (0,0,200)-(180,40,255) doesn't match your court lines")
        print("2. Court lines are not white or have poor contrast")
        print("3. Frame lighting conditions affect detection")
        print("\nTroubleshooting:")
        print("- Check if your court lines are actually white/very light colored")
        print("- Increase HSV_lower[2] (brightness threshold) if lines are grayish")
        print("- Decrease HSV_upper[1] (saturation threshold) if lines are slightly colored")
    else:
        print("\n✓ Lines detected successfully!\n")
        for line_name, line_data in sorted(line_mapping.items()):
            coords = line_data["coords"]
            print(f"\n{line_name.upper()}:")
            print(f"  Type: {line_data['type']}")
            print(f"  Coords: ({coords[0]}, {coords[1]}) → ({coords[2]}, {coords[3]})")
            print(f"  Length: {line_data['length']:.1f} px")
    
    # Save debug frames
    output_dir = STAGE_DIRS[0] / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save annotated frame
    output_path = output_dir / f"debug_frame_{frame_num}.jpg"
    cv2.imwrite(str(output_path), annotated_frame)
    print(f"\n✓ Annotated frame saved: {output_path}")
    
    # Save white mask for inspection
    white_mask = detector.detect_white_mask(frame)
    mask_path = output_dir / f"white_mask_{frame_num}.jpg"
    cv2.imwrite(str(mask_path), white_mask)
    print(f"✓ White mask saved: {mask_path}")
    
    # Save edges for inspection
    morph_mask = detector.apply_morphological_operations(white_mask)
    edges = detector.detect_edges(morph_mask)
    edges_path = output_dir / f"edges_{frame_num}.jpg"
    cv2.imwrite(str(edges_path), edges)
    print(f"✓ Canny edges saved: {edges_path}")
    
    # Save line coordinates
    if line_mapping:
        detector.save_line_coordinates(line_mapping, output_dir)
        print(f"✓ Line coordinates saved to court_lines.json")
    
    print(f"\n{'='*60}")
    print("Debug files saved in: outputs/stage0_court_lines/debug/")
    print("Inspect these images to diagnose detection issues:")
    print("  - white_mask_*.jpg: Shows what's detected as white")
    print("  - edges_*.jpg: Shows edges after Canny detection")
    print("  - debug_frame_*.jpg: Shows final detected lines")
    print(f"{'='*60}\n")
    
    return len(line_mapping) > 0


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_court_lines.py <video_path> [frame_number]")
        print("\nExample:")
        print("  python test_court_lines.py match.mp4")
        print("  python test_court_lines.py match.mp4 100")
        sys.exit(1)
    
    video_path = sys.argv[1]
    frame_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    success = test_court_line_detection(video_path, frame_num)
    sys.exit(0 if success else 1)

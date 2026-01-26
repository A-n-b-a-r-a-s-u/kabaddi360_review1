"""
Visualization utilities for annotating videos and creating UI-friendly outputs.
Handles drawing boxes, skeletons, status overlays, and risk indicators.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from config.config import VIS_CONFIG, RISK_FUSION_CONFIG


class VideoAnnotator:
    """Main class for annotating video frames with detection results."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.colors = VIS_CONFIG["colors"]
        self.font = VIS_CONFIG["font"]
        self.font_scale = VIS_CONFIG["font_scale"]
        self.font_thickness = VIS_CONFIG["font_thickness"]
        self.box_thickness = VIS_CONFIG["box_thickness"]
    
    def draw_bbox(
        self, 
        frame: np.ndarray, 
        bbox: List[int], 
        label: str = "", 
        color: Tuple[int, int, int] = None,
        thickness: int = None
    ) -> np.ndarray:
        """Draw bounding box with label."""
        if color is None:
            color = self.colors["player_box"]
        if thickness is None:
            thickness = self.box_thickness
        
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        if label:
            label_size, _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
            y1_label = max(y1, label_size[1] + 10)
            cv2.rectangle(
                frame, 
                (x1, y1_label - label_size[1] - 10), 
                (x1 + label_size[0], y1_label), 
                color, 
                -1
            )
            cv2.putText(
                frame, 
                label, 
                (x1, y1_label - 5), 
                self.font, 
                self.font_scale, 
                (255, 255, 255), 
                self.font_thickness
            )
        
        return frame
    
    def draw_skeleton(
        self, 
        frame: np.ndarray, 
        landmarks: Dict[str, Tuple[int, int]], 
        connections: List[Tuple[str, str]] = None
    ) -> np.ndarray:
        """Draw pose skeleton from landmarks."""
        if connections is None:
            # Default skeleton connections
            connections = [
                ("nose", "left_shoulder"), ("nose", "right_shoulder"),
                ("left_shoulder", "right_shoulder"),
                ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
                ("left_hip", "right_hip"),
                ("left_hip", "left_knee"), ("right_hip", "right_knee"),
                ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
            ]
        
        color = self.colors["pose_skeleton"]
        
        # Draw connections
        for start_joint, end_joint in connections:
            if start_joint in landmarks and end_joint in landmarks:
                start_pt = tuple(map(int, landmarks[start_joint]))
                end_pt = tuple(map(int, landmarks[end_joint]))
                cv2.line(frame, start_pt, end_pt, color, 2)
        
        # Draw joints
        for joint, (x, y) in landmarks.items():
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
            cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), 1)
        
        return frame
    
    def draw_trajectory(
        self, 
        frame: np.ndarray, 
        points: List[Tuple[int, int]], 
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw motion trajectory."""
        if len(points) < 2:
            return frame
        
        for i in range(len(points) - 1):
            cv2.line(frame, tuple(map(int, points[i])), tuple(map(int, points[i + 1])), color, thickness)
        
        return frame
    
    def draw_risk_meter(
        self, 
        frame: np.ndarray, 
        risk_score: float, 
        position: Tuple[int, int] = (50, 50),
        size: Tuple[int, int] = (300, 60)
    ) -> np.ndarray:
        """Draw risk score meter with color-coded levels."""
        x, y = position
        width, height = size
        
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
        
        # Risk level
        if risk_score <= 30:
            level = "LOW"
            color = VIS_CONFIG["risk_colors"]["low"]
        elif risk_score <= 70:
            level = "MEDIUM"
            color = VIS_CONFIG["risk_colors"]["medium"]
        else:
            level = "HIGH"
            color = VIS_CONFIG["risk_colors"]["high"]
        
        # Fill bar
        fill_width = int((risk_score / 100.0) * (width - 10))
        cv2.rectangle(frame, (x + 5, y + 5), (x + 5 + fill_width, y + height - 5), color, -1)
        
        # Text
        text = f"INJURY RISK: {int(risk_score)}% - {level}"
        text_size, _ = cv2.getTextSize(text, self.font, 0.7, 2)
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), self.font, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def draw_event_alert(
        self, 
        frame: np.ndarray, 
        event_type: str, 
        position: Tuple[int, int] = None,
        duration: int = 30
    ) -> np.ndarray:
        """Draw alert for detected events (fall, collision, etc.)."""
        if position is None:
            position = (self.width // 2 - 150, self.height - 100)
        
        x, y = position
        
        # Alert box
        alert_text = f"{event_type.lower()} detected ?"
        text_size, _ = cv2.getTextSize(alert_text, self.font, 0.8, 2)
        
        box_width = text_size[0] + 40
        box_height = text_size[1] + 30
        
        # Flashing effect
        if (duration % 10) < 5:
            cv2.rectangle(
                frame, 
                (x - 20, y - 15), 
                (x + box_width, y + box_height), 
                (0, 0, 255), 
                -1
            )
            cv2.rectangle(
                frame, 
                (x - 20, y - 15), 
                (x + box_width, y + box_height), 
                (255, 255, 255), 
                3
            )
            cv2.putText(
                frame, 
                alert_text, 
                (x, y + text_size[1]), 
                self.font, 
                0.8, 
                (255, 255, 255), 
                2
            )
        
        return frame
    
    def draw_stats_panel(
        self, 
        frame: np.ndarray, 
        stats: Dict[str, Any],
        position: Tuple[int, int] = (10, 10)
    ) -> np.ndarray:
        """Draw statistics panel with multiple metrics."""
        x, y = position
        line_height = 25
        
        # Background panel
        panel_height = len(stats) * line_height + 20
        cv2.rectangle(frame, (x, y), (x + 350, y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y), (x + 350, y + panel_height), (255, 255, 255), 2)
        
        # Draw stats
        current_y = y + 20
        for key, value in stats.items():
            # Convert numpy types to Python native types
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            elif isinstance(value, np.ndarray):
                value = value.item() if value.size == 1 else str(value)
            
            text = f"{key}: {value}"
            cv2.putText(frame, text, (x + 10, current_y), self.font, 0.5, (255, 255, 255), 1)
            current_y += line_height
        
        return frame
    
    def draw_raider_highlight(
        self, 
        frame: np.ndarray, 
        bbox: List[int],
        glow_intensity: int = 30
    ) -> np.ndarray:
        """Highlight raider with special visual effect."""
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Draw thick border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        
        # Add corner markers
        corner_size = 20
        cv2.line(frame, (x1, y1), (x1 + corner_size, y1), (0, 255, 255), 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_size), (0, 255, 255), 3)
        cv2.line(frame, (x2, y1), (x2 - corner_size, y1), (0, 255, 255), 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_size), (0, 255, 255), 3)
        cv2.line(frame, (x1, y2), (x1 + corner_size, y2), (0, 255, 255), 3)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_size), (0, 255, 255), 3)
        cv2.line(frame, (x2, y2), (x2 - corner_size, y2), (0, 255, 255), 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_size), (0, 255, 255), 3)
        
        # Add label
        cv2.putText(
            frame, 
            "RAIDER", 
            (x1, y1 - 10), 
            self.font, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        return frame
    
    def create_status_card(
        self, 
        stage_name: str, 
        status: str, 
        details: str = "",
        width: int = 400,
        height: int = 150
    ) -> np.ndarray:
        """Create a status card image for UI display."""
        card = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient - improved colors
        for i in range(height):
            intensity = int(40 + (i / height) * 40)
            card[i, :] = [intensity, intensity, intensity + 20]
        
        # Border - thicker and more visible
        cv2.rectangle(card, (0, 0), (width - 1, height - 1), (100, 200, 255), 3)
        
        # Status color
        if status == "Completed":
            status_color = (0, 255, 0)
        elif status == "Processing":
            status_color = (0, 165, 255)
        else:  # Yet to start
            status_color = (128, 128, 128)
        
        # Title - with better positioning
        title_y = 35
        cv2.putText(card, stage_name[:35], (15, title_y), self.font, 0.65, (255, 255, 255), 2)
        
        # Status
        cv2.putText(card, f"Status: {status}", (15, 70), self.font, 0.6, status_color, 2)
        
        # Details - improved text rendering
        if details:
            # Word wrap for details
            words = str(details).split()[:15]
            lines = []
            current_line = []
            for word in words:
                test_line = ' '.join(current_line + [word])
                size, _ = cv2.getTextSize(test_line, self.font, 0.38, 1)
                if size[0] > width - 25:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    current_line.append(word)
            if current_line:
                lines.append(' '.join(current_line))
            
            y_offset = 100
            for line in lines[:2]:
                if line.strip():
                    cv2.putText(card, line[:45], (15, y_offset), self.font, 0.35, (200, 200, 200), 1)
                    y_offset += 18
        
        return card


def create_comparison_view(frames: List[np.ndarray], labels: List[str]) -> np.ndarray:
    """Create side-by-side comparison of multiple frames."""
    if not frames:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Resize all to same size
    target_height = frames[0].shape[0]
    target_width = frames[0].shape[1]
    
    resized_frames = []
    for frame in frames:
        if frame.shape[:2] != (target_height, target_width):
            resized = cv2.resize(frame, (target_width, target_height))
        else:
            resized = frame.copy()
        resized_frames.append(resized)
    
    # Add labels
    for i, (frame, label) in enumerate(zip(resized_frames, labels)):
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Concatenate horizontally
    comparison = np.hstack(resized_frames)
    return comparison

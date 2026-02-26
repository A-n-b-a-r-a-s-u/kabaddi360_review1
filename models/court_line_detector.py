import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from loguru import logger
import json

from config.config import COURT_LINE_CONFIG, STAGE_DIRS


class CourtLineDetector:
    """Detect Kabaddi court lines using HSV-based white line detection."""
    
    def __init__(self):
        """Initialize court line detector."""
        self.detected_lines: List[Dict] = []
        self.line_coordinates: Dict[str, Dict] = {}  # Store as JSON-serializable format
        self.frame_count = 0
        
        logger.info("Court Line Detector initialized")
        logger.info(f"HSV Range: {COURT_LINE_CONFIG['hsv_lower']} to {COURT_LINE_CONFIG['hsv_upper']}")
    
    def detect_white_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame from BGR to HSV and detect white court lines using thresholding.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Binary mask of white lines detected
        """
        logger.debug("[STAGE 0 - COURT LINES] Converting frame to HSV...")
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define HSV range for white color
        lower_hsv = np.array(COURT_LINE_CONFIG["hsv_lower"])
        upper_hsv = np.array(COURT_LINE_CONFIG["hsv_upper"])
        
        # Create mask for white lines
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        logger.debug("[STAGE 0 - COURT LINES] White mask created")
        
        return mask
    
    def apply_morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations (closing + dilation) to clean noise and enhance lines.
        
        Args:
            mask: Binary mask from HSV thresholding
            
        Returns:
            Cleaned mask with enhanced lines
        """
        logger.debug("[STAGE 0 - COURT LINES] Applying morphological operations...")
        
        # Create morphological kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            COURT_LINE_CONFIG["morph_kernel_size"]
        )
        
        # Closing: Dilation followed by Erosion (fills small holes)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                                  iterations=COURT_LINE_CONFIG["morph_iterations"])
        
        # Dilation: Expand white regions (strengthen lines)
        dilated = cv2.dilate(closed, kernel, iterations=COURT_LINE_CONFIG["morph_iterations"])
        
        logger.debug("[STAGE 0 - COURT LINES] Morphological operations complete")
        return dilated
    
    def detect_edges(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply Canny edge detection on the mask to find line edges.
        
        Args:
            mask: Morphologically processed mask
            
        Returns:
            Edge map from Canny edge detection
        """
        logger.debug("[STAGE 0 - COURT LINES] Applying Canny edge detection...")
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(
            blurred,
            COURT_LINE_CONFIG["canny_low"],
            COURT_LINE_CONFIG["canny_high"]
        )
        
        logger.debug("[STAGE 0 - COURT LINES] Canny edge detection complete")
        return edges
    
    def detect_lines(self, edges: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Use Hough Line Transform to detect straight lines in the edge map.
        
        Args:
            edges: Edge map from Canny detection
            
        Returns:
            List of detected lines as [(x1, y1, x2, y2), ...]
        """
        logger.debug("[STAGE 0 - COURT LINES] Applying Hough Line Transform...")
        
        # Hough Probabilistic Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=COURT_LINE_CONFIG["hough_rho"],
            theta=np.radians(COURT_LINE_CONFIG["hough_theta"]),
            threshold=COURT_LINE_CONFIG["hough_threshold"],
            minLineLength=COURT_LINE_CONFIG["min_line_length"],
            maxLineGap=COURT_LINE_CONFIG["max_line_gap"]
        )
        
        detected_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                detected_lines.append((int(x1), int(y1), int(x2), int(y2)))
            
            logger.debug(f"[STAGE 0 - COURT LINES] Detected {len(detected_lines)} raw lines")
        else:
            logger.debug("[STAGE 0 - COURT LINES] No lines detected in frame")
        
        return detected_lines
    
    def classify_lines(self, detected_lines: List[Tuple[int, int, int, int]]) -> Dict[str, Dict]:
        """
        Classify lines as horizontal or vertical, then categorize them as line_1, line_2, line_3, line_4.
        Lines are ordered left-to-right for vertical lines and top-to-bottom for horizontal lines.
        
        Args:
            detected_lines: List of detected lines from Hough transform
            
        Returns:
            Dictionary with classified lines and their properties
        """
        logger.debug("[STAGE 0 - COURT LINES] Classifying detected lines...")
        
        if not detected_lines:
            logger.debug("[STAGE 0 - COURT LINES] No lines to classify")
            return {"vertical_lines": [], "horizontal_lines": [], "line_mapping": {}}
        
        vertical_lines = []
        horizontal_lines = []
        
        angle_tolerance = COURT_LINE_CONFIG["angle_tolerance"]
        
        # Separate horizontal and vertical lines based on angle
        for x1, y1, x2, y2 in detected_lines:
            dx = x2 - x1
            dy = y2 - y1
            
            # Calculate angle (0° = horizontal, 90° = vertical)
            if dx == 0 and dy == 0:
                continue
            
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            
            # Normalize angle to 0-90 range
            if angle > 90:
                angle = 180 - angle
            
            if angle < angle_tolerance or angle > (90 - angle_tolerance):
                # Horizontal line
                horizontal_lines.append({
                    "coords": (x1, y1, x2, y2),
                    "y_center": (y1 + y2) // 2,
                    "x_min": min(x1, x2),
                    "x_max": max(x1, x2),
                    "length": np.sqrt(dx**2 + dy**2)
                })
            elif angle_tolerance < angle < (90 - angle_tolerance):
                # Vertical line
                vertical_lines.append({
                    "coords": (x1, y1, x2, y2),
                    "x_center": (x1 + x2) // 2,
                    "y_min": min(y1, y2),
                    "y_max": max(y1, y2),
                    "length": np.sqrt(dx**2 + dy**2)
                })
        
        logger.debug(f"[STAGE 0 - COURT LINES] Found {len(horizontal_lines)} horizontal, {len(vertical_lines)} vertical lines")
        
        # Merge nearby lines and select most prominent ones
        merged_vertical = self._merge_nearby_lines(vertical_lines, is_vertical=True)
        merged_horizontal = self._merge_nearby_lines(horizontal_lines, is_vertical=False)
        
        # Sort vertical lines left-to-right and horizontal lines top-to-bottom
        merged_vertical.sort(key=lambda l: l["x_center"])
        merged_horizontal.sort(key=lambda l: l["y_center"])
        
        # Assign names: line_1, line_2, line_3, line_4
        line_mapping = {}
        line_count = 1
        
        # Add vertical lines first (left to right)
        for line_idx, line in enumerate(merged_vertical):
            if line_count <= 4:
                line_name = f"line_{line_count}"
                line_mapping[line_name] = {
                    "type": "vertical",
                    "coords": line["coords"],
                    "x_center": line["x_center"],
                    "y_center": (line["y_min"] + line["y_max"]) // 2,
                    "length": line["length"]
                }
                line_count += 1
        
        # Add horizontal lines (top to bottom)
        for line_idx, line in enumerate(merged_horizontal):
            if line_count <= 4:
                line_name = f"line_{line_count}"
                line_mapping[line_name] = {
                    "type": "horizontal",
                    "coords": line["coords"],
                    "x_center": (line["x_min"] + line["x_max"]) // 2,
                    "y_center": line["y_center"],
                    "length": line["length"]
                }
                line_count += 1
        
        logger.info(f"[STAGE 0 - COURT LINES] Classified {len(line_mapping)} major court lines")
        
        return {
            "vertical_lines": merged_vertical,
            "horizontal_lines": merged_horizontal,
            "line_mapping": line_mapping
        }
    
    def _merge_nearby_lines(self, lines: List[Dict], is_vertical: bool) -> List[Dict]:
        """
        Merge lines that are very close to each other (likely the same physical line).
        
        Args:
            lines: List of line dictionaries
            is_vertical: True for vertical lines, False for horizontal
            
        Returns:
            Merged list of lines
        """
        if not lines:
            return []
        
        separation_threshold = COURT_LINE_CONFIG["min_separation"]
        merged = []
        used_indices = set()
        
        for i, line1 in enumerate(lines):
            if i in used_indices:
                continue
            
            cluster = [line1]
            used_indices.add(i)
            
            # Find nearby lines to merge
            for j, line2 in enumerate(lines[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if is_vertical:
                    distance = abs(line1["x_center"] - line2["x_center"])
                else:
                    distance = abs(line1["y_center"] - line2["y_center"])
                
                if distance < separation_threshold:
                    cluster.append(line2)
                    used_indices.add(j)
            
            # Merge cluster - take average
            if is_vertical:
                avg_x = int(np.mean([l["x_center"] for l in cluster]))
                min_y = min([l["y_min"] for l in cluster])
                max_y = max([l["y_max"] for l in cluster])
                merged_line = {
                    "x_center": avg_x,
                    "y_min": min_y,
                    "y_max": max_y,
                    "length": max_y - min_y,
                    "coords": (avg_x, min_y, avg_x, max_y)
                }
            else:
                avg_y = int(np.mean([l["y_center"] for l in cluster]))
                min_x = min([l["x_min"] for l in cluster])
                max_x = max([l["x_max"] for l in cluster])
                merged_line = {
                    "y_center": avg_y,
                    "x_min": min_x,
                    "x_max": max_x,
                    "length": max_x - min_x,
                    "coords": (min_x, avg_y, max_x, avg_y)
                }
            
            merged.append(merged_line)
        
        return merged
    
    def draw_lines(self, frame: np.ndarray, line_mapping: Dict[str, Dict]) -> np.ndarray:
        """
        Draw classified court lines on frame with color coding and labels.
        
        Line color mapping:
        - line_1: Blue
        - line_2: Green
        - line_3: Red
        - line_4: Yellow
        
        Args:
            frame: Input frame
            line_mapping: Dictionary of classified lines
            
        Returns:
            Annotated frame with drawn lines
        """
        annotated_frame = frame.copy()
        
        if not line_mapping:
            logger.debug("[STAGE 0 - COURT LINES] No lines to draw")
            return annotated_frame
        
        colors = COURT_LINE_CONFIG["line_colors"]
        thickness = COURT_LINE_CONFIG["line_thickness"]
        font_scale = COURT_LINE_CONFIG["label_font_scale"]
        font_thickness = COURT_LINE_CONFIG["label_font_thickness"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        logger.debug("[STAGE 0 - COURT LINES] Drawing classified lines on frame...")
        
        # Draw each classified line
        for line_name in sorted(line_mapping.keys(), key=lambda x: int(x.split("_")[1])):
            line_data = line_mapping[line_name]
            x1, y1, x2, y2 = line_data["coords"]
            
            # Get color for this line
            color = colors.get(line_name, (255, 255, 255))  # Default white
            
            # Draw line
            cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Draw label
            label_x = int((x1 + x2) // 2)
            label_y = int((y1 + y2) // 2)
            
            # Offset label slightly to avoid overlap with line
            label_y_offset = 20
            
            cv2.putText(
                annotated_frame,
                line_name.upper(),
                (label_x, label_y - label_y_offset),
                font,
                font_scale,
                color,
                font_thickness
            )
        
        # Add summary
        summary_text = f"Court Lines Detected: {len(line_mapping)}"
        cv2.putText(
            annotated_frame,
            summary_text,
            (10, 30),
            font,
            0.7,
            (255, 255, 255),
            2
        )
        
        logger.debug(f"[STAGE 0 - COURT LINES] Drew {len(line_mapping)} lines on frame")
        
        return annotated_frame
    
    def save_line_coordinates(self, line_mapping: Dict[str, Dict], output_dir: Path):
        """
        Save detected court line coordinates to JSON file for use by downstream stages.
        
        Args:
            line_mapping: Dictionary of classified lines
            output_dir: Output directory for JSON file
        """
        if not line_mapping:
            logger.debug("[STAGE 0 - COURT LINES] No lines to save")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        court_lines_data = {}
        for line_name, line_data in line_mapping.items():
            coords = line_data["coords"]
            court_lines_data[line_name] = {
                "type": line_data.get("type", "unknown"),
                "coords": {
                    "x1": int(coords[0]),
                    "y1": int(coords[1]),
                    "x2": int(coords[2]),
                    "y2": int(coords[3])
                },
                "center": {
                    "x": int(line_data.get("x_center", 0)),
                    "y": int(line_data.get("y_center", 0))
                },
                "length": float(line_data.get("length", 0))
            }
        
        # Save to JSON
        json_path = output_dir / "court_lines.json"
        with open(json_path, 'w') as f:
            json.dump(court_lines_data, f, indent=2)
        
        logger.info(f"[STAGE 0 - COURT LINES] Court line coordinates saved to {json_path}")
    
    def get_raider_detection_line(self, frame_width: int) -> int:
        """
        Get the x-coordinate of the 25% reference line for raider detection.
        Players crossing this line from left to right are marked as raiders.
        
        Args:
            frame_width: Width of the frame
            
        Returns:
            x-coordinate of the 25% line
        """
        return int(frame_width * 0.25)
    
    def draw_raider_reference_line(self, annotated_frame: np.ndarray) -> np.ndarray:
        """
        Draw the 25% reference line for raider detection on the frame.
        
        Args:
            annotated_frame: Frame with court lines already drawn
            
        Returns:
            Frame with reference line added
        """
        height, width = annotated_frame.shape[:2]
        reference_x = self.get_raider_detection_line(width)
        
        # Draw reference line in cyan (bright color)
        color = (255, 255, 0)  # Cyan in BGR
        thickness = 3
        cv2.line(annotated_frame, (reference_x, 0), (reference_x, height), color, thickness)
        
        # Add label
        cv2.putText(
            annotated_frame,
            "RAIDER LINE (25%)",
            (reference_x - 50, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        logger.debug(f"[STAGE 0 - COURT LINES] Raider reference line drawn at x={reference_x}")
        
        return annotated_frame
    
    def process_frame(self, frame: np.ndarray, frame_num: int = 0, save_intermediate: bool = False, output_dir: Path = None) -> Tuple[np.ndarray, Dict]:
        """
        Process single frame through complete court line detection pipeline.
        
        Pipeline:
        1. Convert BGR to HSV and threshold for white lines
        2. Apply morphological operations (closing + dilation)
        3. Apply Canny edge detection
        4. Use Hough Line Transform to detect lines
        5. Classify and categorize lines
        6. Draw lines on frame
        7. Draw raider reference line (25% from left)
        
        Args:
            frame: Input frame in BGR format
            frame_num: Frame number for logging
            save_intermediate: Whether to save intermediate processing steps
            output_dir: Directory to save intermediate images
            
        Returns:
            Tuple of (annotated_frame, line_mapping)
        """
        logger.debug(f"[STAGE 0 - COURT LINES] Processing frame {frame_num}...")
        
        self.frame_count += 1
        
        # Step 1: Detect white mask
        white_mask = self.detect_white_mask(frame)
        if save_intermediate and output_dir and frame_num in [0, 100, 200]:  # Save only 3 key frames
            cv2.imwrite(str(output_dir / f"{frame_num:06d}_01_greyscale.jpg"), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            cv2.imwrite(str(output_dir / f"{frame_num:06d}_02_blurred.jpg"), cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0))
        
        # Step 2: Apply morphological operations
        morph_mask = self.apply_morphological_operations(white_mask)
        
        # Step 3: Canny edge detection
        edges = self.detect_edges(morph_mask)
        if save_intermediate and output_dir and frame_num in [0, 100, 200]:  # Save for 3 key frames
            cv2.imwrite(str(output_dir / f"{frame_num:06d}_03_canny_edges.jpg"), edges)
        
        # Step 4: Hough Line Transform
        detected_lines = self.detect_lines(edges)
        
        # Save hough line visualization
        if save_intermediate and output_dir and frame_num in [0, 100, 200]:  # Save for 3 key frames
            hough_vis = self._visualize_hough_lines(edges, detected_lines, frame.shape)
            cv2.imwrite(str(output_dir / f"{frame_num:06d}_04_hough_lines.jpg"), hough_vis)
        
        # Step 5: Classify lines
        classification_result = self.classify_lines(detected_lines)
        line_mapping = classification_result["line_mapping"]
        
        # Store for later access
        self.detected_lines = [line_mapping[name] for name in sorted(line_mapping.keys())]
        self.line_coordinates = line_mapping
        
        # Step 6: Draw lines on frame
        annotated_frame = self.draw_lines(frame, line_mapping)
        
        # Step 7: Draw raider reference line
        annotated_frame = self.draw_raider_reference_line(annotated_frame)
        
        logger.debug(f"[STAGE 0 - COURT LINES] Frame {frame_num}: Detected {len(line_mapping)} court lines")
        
        return annotated_frame, line_mapping
    
    def _visualize_hough_lines(self, edges: np.ndarray, lines: List[Tuple], frame_shape: Tuple) -> np.ndarray:
        """
        Visualize Hough lines on a black background.
        
        Args:
            edges: Edge map
            lines: Detected lines from Hough transform
            frame_shape: Shape of original frame
            
        Returns:
            Visualization image with white lines on black background
        """
        # Create black canvas
        vis = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
        
        if lines:
            for x1, y1, x2, y2 in lines:
                cv2.line(vis, (x1, y1), (x2, y2), 255, 2)
        
        return vis
    
    def get_line_coordinates(self) -> Dict[str, Dict]:
        """
        Get the last detected line coordinates.
        
        Returns:
            Dictionary of line coordinates
        """
        return self.line_coordinates
    
    def reset(self):
        """Reset detector state."""
        self.detected_lines = []
        self.line_coordinates = {}
        self.frame_count = 0
        logger.info("Court Line Detector reset")


if __name__ == "__main__":
    logger.info("Court Line Detector module loaded")

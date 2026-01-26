# visualization.py - Documentation

## 1. Overview
`utils/visualization.py` is the extensive drafting library for the project. It centralizes all visual logic—colors, fonts, layout—ensuring that every output (Annotated Video, Status Cards, Charts) has a consistent, professional "Kabaddi Tech" aesthetic.

## 2. Technical Overview
- **Type**: Graphics / UI Rendering Module
- **Dependencies**: `cv2`, `numpy`, `config.VIS_CONFIG`.
- **Design System**: Uses a centralized config for colors (e.g., specific Neon Green for safe, Red for danger) to maintain visual consistency.

## 3. Detailed Technical
This module performs all "drawing" operations on NumPy arrays.
- **Smart Text**: Measures text size before drawing background rectangles, ensuring labels are always readable regardless of background contrast.
- **Gauges**: Implements custom UI elements like the "Risk Meter" (a progress bar drawn with OpenCV primitives).
- **Skeleton Rendering**: Connects keypoints with lines to visualize the pose.
- **Alert System**: unique "Flashing" effect logic for critical events (Fall/Impact) to grab attention.

## 4. Functions

### `VideoAnnotator` Class

##### `draw_bbox(...)`
- **Purpose**: Object labeling.
- **Details**: Draws standardized box + text label with background pill.

##### `draw_skeleton(frame, landmarks)`
- **Purpose**: Pose viz.
- **Details**: Iterates connection pairs (e.g., Shoulder-Elbow) and lines them. Renders joints as two-tone circles.

##### `draw_risk_meter(frame, score)`
- **Purpose**: HUD element.
- **Details**: Renders a bar at top-left. Color changes dynamically (Green->Orange->Red) based on score.

##### `draw_event_alert(frame, event_type)`
- **Purpose**: Notification.
- **Details**: Draws a large flashing warning box at the bottom center.

##### `create_status_card(...)`
- **Purpose**: UI Asset Generation.
- **Details**: Creates a standalone image file (not an overlay) used by the Streamlit frontend to display stage progress cards. Uses gradient backgrounds.

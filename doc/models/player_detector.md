# player_detector.py - Documentation

## 1. Overview
`models/player_detector.py` implements **Stage 1** of the pipeline. Its primary responsibility is to detect all human figures in the video frame and assign them unique, consistent IDs across frames. This forms the foundation for all subsequent analysis; if detections fail here, no other stage can function.

## 2. Technical Overview
- **Type**: Deep Learning Detection + Tracking Module
- **Dependencies**: 
    - `ultralytics`: For the YOLOv8 object detection model.
    - `cv2`, `numpy`: Image processing.
    - `utils.tracking_utils.SORTTracker`: Implementation of the SORT (Simple Online and Realtime Tracking) algorithm.
- **Model**: YOLOv8 Nano (`yolov8n.pt`) is used by default for speed, detecting class `0` (person).
- **Input**: Raw video frame (numpy array).
- **Output**: 
    - `annotated_frame`: Frame with bounding boxes drawn.
    - `players`: List of dictionaries, each containing `track_id`, `bbox` (x1, y1, x2, y2), and `center` point.

## 3. Detailed Technical
The module combines a state-of-the-art detector with a classic tracker.
- **Detection**: YOLOv8 scans the frame. It is configured with a confidence threshold (default 0.5) to filter weak detections. It runs on GPU (`device='cuda'`) if available.
- **Tracking**: The SORT algorithm matches detections in the current frame to existing tracks from previous frames based on IOU (Intersection over Union). It handles ID assignment and recovery for brief occlusions (up to 30 frames).
- **Trajectory**: Maintains a short history of center points (last 30 frames) to visualize movement paths.

## 4. Workflow and Function Details

### Workflow
1.  **Initialize**: Load YOLO model and initialize SORT tracker.
2.  **Detect**: Run model inference on frame to get bounding boxes.
3.  **Track**: Feed boxes into SORT to get IDs.
4.  **Format**: Convert tracker output (array) into structured dictionaries for the pipeline.
5.  **Annotate**: Draw visual feedback.

### Functions

#### `PlayerDetector` Class

##### `__init__(self)`
- **Purpose**: Constructor.
- **Details**: Loads `yolov8n.pt` using `ultralytics.YOLO`. Sets up `SORTTracker` with config values (max_age, min_hits).

##### `detect_players(self, frame) -> np.ndarray`
- **Purpose**: Raw object detection.
- **Details**: Runs `self.model(frame)`. Filters results to keep only class 0 (person). Returns array of shape `(N, 5)`: `[x1, y1, x2, y2, confidence]`.

##### `update_tracks(self, detections) -> np.ndarray`
- **Purpose**: Track management.
- **Details**: Calls `self.tracker.update()`. Updates internal `track_history` dictionary for trajectory visualization.

##### `process_frame(self, frame, frame_num) -> (frame, list)`
- **Purpose**: Main processing wrapper.
- **Details**: Orchestrates detection -> tracking -> formatting. Returns the tuple `(annotated_frame, players_list)`.

##### `annotate_frame(self, frame, tracks)`
- **Purpose**: Visualization.
- **Details**: Draws green bounding boxes, IDs, and cyan trajectory lines using `utils.visualization.VideoAnnotator`.

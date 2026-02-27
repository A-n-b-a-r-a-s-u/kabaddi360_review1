# Kabaddi Injury Prediction System - Comprehensive Technical Documentation

**Project:** Kabaddi Injury Prediction System
**Version:** 1.0
**Date:** February 27, 2026
**Scope:** Complete 7-Stage Pipeline Architecture with Data Flow Analysis

---

## Table of Contents

1. System Overview
2. Data Flow Architecture
3. Stage-by-Stage Implementation Details
4. Data Structure Specifications
5. Integration Points and Communication
6. Performance Metrics and Outputs

---

## 1. System Overview

### 1.1 Project Objective

The Kabaddi Injury Prediction System is a video-only artificial intelligence solution that analyzes Kabaddi match videos to predict injury risk in real-time. The system focuses on identifying and monitoring the raider while analyzing defender interactions, fall events, biomechanical anomalies, and impact severity.

### 1.2 System Architecture

```
INPUT VIDEO
    |
    v
[STAGE 0] Court Line Detection
    |
    v
[STAGE 1] Player Detection & Tracking
    |
    v
[STAGE 2] Raider Identification
    |
    v
[STAGE 3] Pose Estimation (Raider Only)
    |
    v
[STAGE 4] Fall Detection
    |
    v
[STAGE 5] Motion Analysis
    |
    v
[STAGE 6] Impact Detection
    |
    v
[STAGE 7] Risk Fusion & Final Score
    |
    v
OUTPUT: Annotated Video + Risk Metrics + Event Timeline
```

### 1.3 Key Components

- **Video Processing Engine:** OpenCV-based frame processing
- **Detection Models:** YOLOv8 (nano) for player detection
- **Pose Estimation:** MediaPipe for skeleton extraction
- **Tracking Algorithm:** SORT (Simple Online and Realtime Tracking)
- **Temporal Analysis:** LSTM network for motion profiling
- **Risk Calculation:** Weighted multi-factor fusion engine
- **Dashboard:** Streamlit real-time visualization interface

---

## 2. Data Flow Architecture

### 2.1 Complete Data Pipeline

```
VIDEO FILE (MP4/AVI)
    |
    +---> Video Reader (fps, resolution, frame count)
    |
    +===[FRAME STREAM]==+
    |                  |
    +---> Frame (numpy.ndarray: H x W x 3 in BGR)
    |
    +---> Stage 0: Court Lines Detection
    |         Input:  RGB frame
    |         Output: Court line coordinates (JSON)
    |
    +---> Stage 1: Player Detection
    |         Input:  RGB frame + Court lines
    |         Output: Player bounding boxes + Track IDs
    |
    +---> Stage 2: Raider Identification
    |         Input:  Player positions + Track history
    |         Output: Raider track ID
    |
    +---> Stage 3: Pose Estimation
    |         Input:  Frame + Raider bbox
    |         Output: Joint positions (17 landmarks)
    |
    +---> Stage 4: Fall Detection
    |         Input:  Joint positions + Historical data
    |         Output: Fall event + Severity score
    |
    +---> Stage 5: Motion Analysis
    |         Input:  Joint positions + Historical data
    |         Output: Motion abnormality score
    |
    +---> Stage 6: Impact Detection
    |         Input:  All player positions + Raider ID
    |         Output: Collision events + Impact severity
    |
    +---> Stage 7: Risk Fusion
    |         Input:  All component scores
    |         Output: Final risk score (0-100)
    |
    +---> Frame Annotation
    |         Input:  Frame + All metrics + Raider info
    |         Output: Annotated frame
    |
    +---> Video Writer
    |         Output: Final MP4 video
    |
    +---> Metrics Export
            Output: JSON files (metrics, events, summary)
```

### 2.2 Data Transformation at Each Stage

| Stage | Input Format | Processing | Output Format |
|-------|--------------|-----------|---------------|
| 0 | BGR Frame (H×W×3) | HSV threshold + morphology + Hough | List[Tuple(x1,y1,x2,y2)] |
| 1 | BGR Frame | YOLO detection + SORT tracking | List[Dict{bbox, track_id, center}] |
| 2 | Player list + Track history | Line crossing detection + temporal consistency | int (raider_track_id) |
| 3 | Frame + Raider bbox | MediaPipe pose + Kalman filtering | Dict{joint: {position, velocity}} |
| 4 | Joint positions + History | Biomechanical rule-based analysis | Dict{fall_bool, severity, indicators} |
| 5 | Joint positions + History | Velocity/acceleration/jerk + LSTM | Dict{abnormality_score, metrics} |
| 6 | All players + Raider ID | Collision detection + approach angle analysis | Dict{impact_bool, severity, defenders} |
| 7 | All component scores + History | Weighted fusion + temporal smoothing | Dict{risk_score, level, breakdown} |

### 2.3 Data Persistence and State Management

```
Session State (per video processing):
├── PipelineStatus (stage completion tracking)
├── MetricsLogger (frame-by-frame metrics)
├── EventLogger (real-time event stream)
├── ValueLogger (intermediate step values)
├── RaiderStatusTracker (raider identification state)
└── CollisionHistory (accumulated collision events)

Output Files Generated:
├── final_output.mp4 (annotated video)
├── pipeline_summary.json (complete analysis)
├── metrics.json (frame-level metrics)
├── events_timeline.json (temporal event sequence)
├── collision_data.json (collision events)
├── pipeline_status.json (stage completion status)
└── logs/ (detailed debug logs)
```

---

## 3. Stage-by-Stage Implementation Details

---

### Stage 0: Court Line Detection

#### 3.0.1 Objective

Detect and classify the four main lines of a Kabaddi court to establish spatial reference coordinates for raider entry detection and team classification.

#### 3.0.2 Precise Implementation Steps

**Step 1: Color Space Conversion**
- Input: BGR frame from video
- Process: Convert BGR to HSV color space using OpenCV
- Rationale: HSV provides better color invariance under lighting variations
- Output: HSV frame (same dimensions as input)

**Step 2: White Line Mask Creation**
- Input: HSV frame
- Threshold Values:
  - Lower boundary: (0, 0, 200) - H, S, V
  - Upper boundary: (180, 40, 255) - H, S, V
- Process: Create binary mask using cv2.inRange()
- Output: Binary mask (0=not white, 255=white line)
- Quality Metric: Pixel count in mask

**Step 3: Morphological Operations**
- Input: Binary mask
- Process Sequence:
  1. Closing (cv2.MORPH_CLOSE): Dilation + Erosion
     - Kernel size: 5×5
     - Iterations: 2
     - Purpose: Fill small holes and gaps
  2. Dilation (cv2.MORPH_DILATE):
     - Kernel size: 5×5
     - Iterations: 2
     - Purpose: Strengthen and expand thin lines
- Output: Enhanced binary mask
- Effect: Removes noise and creates continuous line segments

**Step 4: Edge Detection**
- Input: Morphologically processed mask
- Preprocessing: Gaussian blur (kernel 5×5, sigma 0)
- Canny Edge Detection Parameters:
  - Low threshold: 50
  - High threshold: 150
- Process: Edge pixels are identified at intensity gradients
- Output: Edge map (binary, only edges highlighted)

**Step 5: Line Detection via Hough Transform**
- Input: Edge map
- Algorithm: Probabilistic Hough Line Transform (cv2.HoughLinesP)
- Parameters:
  - Rho (distance resolution): 1 pixel
  - Theta (angle resolution): 1 degree
  - Threshold: 50 votes minimum
  - Minimum line length: 100 pixels
  - Maximum line gap: 10 pixels
- Output: List of detected lines as [(x1, y1, x2, y2), ...]
- Quantity: Typically 4-12 initial line detections

**Step 6: Line Classification**
- Input: Raw detected lines
- Classification Process:
  1. Calculate angle for each line: arctan((y2-y1)/(x2-x1))
  2. Identify vertical lines (angle ≈ 90°, tolerance ±15°)
  3. Identify horizontal lines (angle ≈ 0°, tolerance ±15°)
  4. Sort vertical lines by x-coordinate (left to right)
  5. Sort horizontal lines by y-coordinate (top to bottom)
  6. Assign numerical labels: line_1, line_2, line_3, line_4
- Minimum separation filter: 50 pixels between adjacent lines
- Output: Dictionary with line classifications and coordinates

**Step 7: Coordinate Extraction and Storage**
- Extract precise endpoints for each classified line
- Store in JSON-serializable format:
  ```
  {
    "line_1": {"x1": int, "y1": int, "x2": int, "y2": int, "angle": float},
    "line_2": {...},
    "line_3": {...},
    "line_4": {...}
  }
  ```
- Save to: `outputs/stage0_court_lines/court_lines.json`

#### 3.0.3 Key Thresholds and Configuration

| Parameter | Value | Impact |
|-----------|-------|--------|
| HSV S upper bound | 40 | Selects bright whites (low saturation) |
| HSV V lower bound | 200 | Selects bright pixels only |
| Canny low threshold | 50 | Minimum edge strength |
| Canny high threshold | 150 | Strong edge confirmation |
| Hough threshold | 50 | Minimum line votes |
| Angle tolerance | 15° | Classification flexibility |

#### 3.0.4 Data Output from Stage 0

```json
{
  "frame_number": 0,
  "timestamp": 0.0,
  "lines_detected": 4,
  "line_coordinates": {
    "line_1": {"x1": 120, "y1": 50, "x2": 120, "y2": 650, "angle": 90.0},
    "line_2": {"x1": 350, "y1": 50, "x2": 350, "y2": 650, "angle": 90.0},
    "line_3": {"x1": 50, "y1": 300, "x2": 900, "y2": 300, "angle": 0.0},
    "line_4": {"x1": 50, "y1": 400, "x2": 900, "y2": 400, "angle": 0.0}
  },
  "detection_line_x": 225
}
```

---

### Stage 1: Player Detection and Tracking

#### 3.1.1 Objective

Detect all persons in the frame and maintain consistent track identities across frames using motion-based association.

#### 3.1.2 Precise Implementation Steps

**Step 1: YOLO Model Initialization**
- Model: YOLOv8 Nano (yolov8n.pt)
- Configuration:
  - Input size: Dynamic (resized to 640×640 internally)
  - Class filter: 0 (Person class only)
  - Device: CUDA if available, CPU otherwise
  - Output: Detection tensors with bounding boxes and confidence

**Step 2: Frame Preparation**
- Input: BGR frame (H×W×3)
- Preprocessing: YOLOv8 internally handles:
  - RGB conversion
  - Resizing to model input dimensions
  - Normalization to [0, 1] range

**Step 3: Person Detection**
- Input: Preprocessed frame
- YOLO Forward Pass:
  - Process through 8 detection heads
  - Generate bounding box predictions
  - Calculate confidence scores for each detection
- Output: Raw detections (all objects with confidence)
- Typical output: 50-300 detections per frame

**Step 4: Detection Filtering**
- Input: All YOLO detections
- Filters applied:
  1. Class filter: Keep only class 0 (person) - removes other objects
  2. Confidence filter: confidence >= 0.5 (configurable)
  3. IOU filter: Remove overlapping detections (IOU threshold 0.45)
- Output: Filtered detections, typically 5-20 players
- Format: [[x1, y1, x2, y2, confidence], ...]

**Step 5: Track Association (SORT Algorithm)**
- Input: Current frame detections, tracker state
- SORT Tracking Process:
  1. Prediction: Use Kalman filter to predict previous track positions
  2. Association: Match detections to predicted tracks using IoU metric
  3. Update: Update Kalman filter state with matched detections
  4. Creation: Create new tracks for unmatched detections
  5. Termination: Remove tracks without detections for >30 frames
- Association Parameters:
  - Max age: 30 frames (lost track is kept alive for 30 frames)
  - Min hits: 3 (track confirmed after 3 detections)
  - IOU threshold: 0.3 (for association)

**Step 6: Track History Maintenance**
- For each active track:
  - Calculate center point: (x1+x2)/2, (y1+y2)/2
  - Append to trajectory history (deque, max length 30 points)
  - Maintain 5 most recent x-positions for raider detection
- Storage: Dictionary keyed by track_id

**Step 7: Player Data Preparation**
- Extract for each track:
  ```
  {
    "track_id": int,
    "bbox": [x1, y1, x2, y2],
    "center": (cx, cy),
    "frame_num": frame_number,
    "confidence": detection_confidence
  }
  ```
- Generate list of all players in frame

#### 3.1.3 Key Algorithm: SORT Tracking

```
Simplified SORT pseudocode:

for each detection D:
    predicted_tracks = kalman_filter.predict(previous_tracks)
    unmatched_detections = D
    
    for each predicted_track T:
        best_match = find_best_matching_detection(T, unmatched_detections)
        if best_match found (IoU > 0.3):
            kalman_filter.update(T, best_match)
            remove best_match from unmatched_detections
        else:
            mark T as "lost" (but keep updating Kalman)
    
    for each unmatched_detection D:
        if D matches "lost" track within max_age:
            reactivate track
        else:
            create new track with id = next_id++
    
    remove tracks marked for >max_age frames
```

#### 3.1.4 Data Output from Stage 1

```python
players_list = [
    {
        "track_id": 1,
        "bbox": [120.5, 100.2, 250.1, 400.8],
        "center": (185, 250),
        "frame_num": 150,
        "confidence": 0.92
    },
    {
        "track_id": 2,
        "bbox": [300.1, 110.5, 420.2, 390.1],
        "center": (360, 250),
        "frame_num": 150,
        "confidence": 0.88
    },
    # ... more players
]

# Saved to: outputs/stage1_detection/detections_frame_{frame_num}.json
```

#### 3.1.5 Tracking Performance Metrics

| Metric | Target | Effect if Poor |
|--------|--------|-----------------|
| Min hits = 3 | 3+ detections | <3: too many ghost tracks; >5: slow confirmation |
| Max age = 30 | 30 frames | <20: frequent ID switches; >50: ghost tracks persist |
| IOU threshold = 0.3 | IoU > 0.3 for match | <0.2: over-association; >0.5: under-association |

---

### Stage 2: Raider Identification

#### 3.2.1 Objective

Identify and lock the raider (attacking player) based on court position crossing using multi-cue temporal consistency.

#### 3.2.2 Precise Implementation Steps

**Step 1: Court Line Detection (from Stage 0)**
- Input: Court line coordinates from Stage 0
- Extract: Raider detection line x-coordinate = 25% from left edge
- Formula: detection_line_x = frame_width × 0.25
- Purpose: Marks the defensive half / raider entry boundary

**Step 2: Player Position History Initialization**
- Input: Player detections from Stage 1
- For each track_id:
  - Initialize position history: deque[int] (x-coordinates)
  - Store current frame center x-coordinate
  - Maintain last 5 positions for crossing detection
- Update process: Each new frame, append current x to history

**Step 3: Crossing Detection**
- Input: Player track history, detection line x-coordinate
- Algorithm:
  ```
  for each player with track_id T:
      if raider_locked == False:
          get historical positions: pos[0], pos[1], ..., pos[-1]
          if len(positions) >= 2:
              prev_x = positions[0]  (oldest position)
              curr_x = positions[-1] (newest position)
              
              if prev_x < detection_line_x AND curr_x > detection_line_x:
                  # CROSSING DETECTED!
                  raider_id = T
                  raider_locked = True
                  raider_confidence[T] = 1.0
  ```
- Condition: Player must cross from LEFT to RIGHT across detection line

**Step 4: Raider Confirmation**
- Upon first crossing detection:
  - Set current_raider_id = track_id
  - Lock raider permanently (raider_locked = True)
  - Set confidence to 1.0 (100%)
- Once locked:
  - Ignore all other players
  - Always return current_raider_id for remaining frames
  - Never re-identify unless video ends

**Step 5: Raider Persistence**
- Input: Locked raider_id
- Behavior if raider temporarily disappears (undetected):
  - Keep raider_id locked
  - Raider deactivates when track lost for >30 frames (SORT max_age)
  - If raider track reappears, reactivate with same ID
- Rationale: Raider cannot "change" mid-match; temporary occlusion is acceptable

**Step 6: Confidence Tracking**
- Maintain confidence score for raider_id:
  - Initial crossing: confidence = 1.0
  - Maintained throughout video: confidence = 1.0
  - No uncertainty or degradation after lock
- Alternative players remain at confidence = 0.0

**Step 7: Team Classification (Derived from Raider)**
- With raider_id identified:
  - Raider: track_id == raider_id
  - Attackers: players on left side (x < detection_line_x)
  - Defenders: players on right side (x >= detection_line_x)
- Classification updated each frame based on current positions

#### 3.2.3 Crossing Detection Algorithm Details

```python
detect_raider_by_line_crossing(players: List[Dict]) -> Optional[int]:
    """
    Input: List of player detections with track_id and center position
    Output: raider track_id if crossing detected, None otherwise
    """
    if raider_locked and detected_raider_id is not None:
        return detected_raider_id  # Return locked raider
    
    update_player_positions(players)  # Update position history
    
    for player in players:
        track_id = player['track_id']
        current_x = player['center'][0]
        
        if track_id not in player_positions_history:
            continue
        
        positions = player_positions_history[track_id]
        if len(positions) < 2:
            continue
        
        prev_x = positions[0]  # Oldest stored position
        
        # Check for left-to-right crossing
        if prev_x < DETECTION_LINE_X and current_x > DETECTION_LINE_X:
            detected_raider_id = track_id
            raider_locked = True
            raider_confidence[track_id] = 1.0
            return track_id
    
    return None
```

#### 3.2.4 Key Parameters for Raider Identification

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Detection line | 25% frame width | Arbitrary boundary marking raider entry |
| History length | 5 positions | Sufficient for smooth crossing detection |
| Lock mechanism | Permanent | Once identified, raider cannot change |
| Crossing direction | Left to right | Raider always attacks from own side |

#### 3.2.5 Data Output from Stage 2

```json
{
  "frame_number": 125,
  "raider_id": 7,
  "raider_locked": true,
  "raider_confidence": 1.0,
  "detection_line_x": 225,
  "team_classification": {
    "raider": {"track_id": 7, "center": [400, 300]},
    "attackers": [
      {"track_id": 2, "center": [150, 280]},
      {"track_id": 5, "center": [180, 320]}
    ],
    "defenders": [
      {"track_id": 1, "center": [500, 290]},
      {"track_id": 3, "center": [550, 310]},
      {"track_id": 4, "center": [480, 330]}
    ]
  }
}
```

---

### Stage 3: Pose Estimation (Raider Only)

#### 3.3.1 Objective

Extract 17-point skeleton (joint positions) from the raider to enable biomechanical analysis in subsequent stages.

#### 3.3.2 Precise Implementation Steps

**Step 1: Raider Region Extraction**
- Input: Frame (H×W×3 BGR), raider bounding box [x1, y1, x2, y2]
- Process:
  1. Expand bounding box by 20 pixels margin on all sides
     - x1_expanded = max(0, x1 - 20)
     - y1_expanded = max(0, y1 - 20)
     - x2_expanded = min(W, x2 + 20)
     - y2_expanded = min(H, y2 + 20)
  2. Crop frame: raider_region = frame[y1_exp:y2_exp, x1_exp:x2_exp]
  3. Validate: If crop is empty, return None (pose detection failed)
- Output: Cropped RGB image (crop_H × crop_W × 3)

**Step 2: Color Space Conversion**
- Input: Cropped raider region (BGR)
- Process: cv2.cvtColor(raider_crop, cv2.COLOR_BGR2RGB)
- Output: RGB crop (required by MediaPipe)

**Step 3: MediaPipe Pose Detection**
- Model: MediaPipe Pose (blazepose architecture)
- Configuration:
  - Static image mode: False (optimized for video)
  - Model complexity: 1 (standard, balance speed/accuracy)
  - Smooth landmarks: True (temporal smoothing)
  - Enable segmentation: False (not needed)
  - Min detection confidence: 0.5
  - Min tracking confidence: 0.5
- Input: RGB crop
- Output: MediaPipe results with 33 landmarks (we use 17 key joints)

**Step 4: Landmark Extraction**
- MediaPipe produces 33 landmarks, map to 17 key joints:

| Joint Name | MediaPipe Index | Body Part |
|-----------|-----------------|-----------|
| nose | 0 | Head |
| left_eye | 1 | Head |
| right_eye | 2 | Head |
| left_shoulder | 11 | Upper body |
| right_shoulder | 12 | Upper body |
| left_elbow | 13 | Arm |
| right_elbow | 14 | Arm |
| left_wrist | 15 | Hand |
| right_wrist | 16 | Hand |
| left_hip | 23 | Torso |
| right_hip | 24 | Torso |
| left_knee | 25 | Leg |
| right_knee | 26 | Leg |
| left_ankle | 27 | Leg |
| right_ankle | 28 | Leg |
| left_foot_index | 31 | Foot |
| right_foot_index | 32 | Foot |

**Step 5: Coordinate Transformation**
- Input: Normalized landmark coordinates (0-1) from MediaPipe on crop
- Process:
  ```
  x_pixel = landmark.x * crop_width + x1_expanded
  y_pixel = landmark.y * crop_height + y1_expanded
  ```
- Output: Absolute frame coordinates (x, y) in pixels
- Extract visibility: landmark.visibility (0-1, confidence of detection)

**Step 6: Kalman Filtering for Smoothing**
- Input: Raw joint position (x, y)
- Kalman filter per joint:
  - Process noise: 0.01 (smooth movements)
  - Measurement noise: 0.1 (sensor noise)
  - State: [x, y, vx, vy] (position + velocity)
- Update: smoothed_position = kalman_filter.update([raw_x, raw_y])
- Output: Smoothed joint position (x, y)
- Benefit: Reduces jitter from frame-to-frame noise

**Step 7: Velocity Calculation**
- Input: Kalman filter internal state
- Extract velocity: [vx, vy] from Kalman filter state
- Output: Joint velocity vector (pixels/frame)
- Usage: For motion analysis in Stage 5

**Step 8: Joint History Maintenance**
- For each joint:
  - Append smoothed position to history (deque, max 90 frames)
  - Keep last 3 seconds at 30 fps
- Storage: Dictionary keyed by joint_name

**Step 9: Output Dictionary Construction**
- Build dictionary for all 17 joints:
  ```python
  joints = {
      "nose": {
          "position": [x, y],
          "velocity": [vx, vy],
          "visibility": 0.95
      },
      "left_shoulder": {
          "position": [x, y],
          "velocity": [vx, vy],
          "visibility": 0.98
      },
      # ... 15 more joints
  }
  ```

#### 3.3.3 MediaPipe Configuration Rationale

| Config | Value | Rationale |
|--------|-------|-----------|
| static_image_mode | False | Optimized for temporal consistency |
| model_complexity | 1 | Balance: not as fast as 0, less heavy than 2 |
| smooth_landmarks | True | Built-in temporal smoothing |
| min_detection_conf | 0.5 | Conservative: accept sketchy detections |
| min_tracking_conf | 0.5 | Same as above |

#### 3.3.4 Kalman Filter Parameters

- Process Variance (Q): 0.01 (assumes smooth motion)
- Measurement Variance (R): 0.1 (camera jitter/noise)
- State Vector: [x, y, vx, vy] (position and velocity)
- Update Frequency: Every frame (30 Hz)

#### 3.3.5 Data Output from Stage 3

```python
joints = {
    "nose": {
        "position": [450.2, 180.5],
        "velocity": [2.1, -1.3],
        "visibility": 0.96
    },
    "left_shoulder": {
        "position": [380.1, 320.5],
        "velocity": [1.5, 0.5],
        "visibility": 0.98
    },
    "right_shoulder": {
        "position": [520.3, 318.2],
        "velocity": [1.8, 0.3],
        "visibility": 0.97
    },
    # ... 14 more joints
}

# If pose detection fails: joints = None
# Saved to: outputs/stage3_pose/pose_frame_{frame_num}.json
```

---

### Stage 4: Fall Detection

#### 3.4.1 Objective

Detect falling events and quantify fall severity using biomechanical indicators derived from joint positions and motion.

#### 3.4.2 Precise Implementation Steps

**Step 1: Joint History Initialization**
- Input: Previous joint data
- Maintain circular buffers for:
  - Hip positions (deque, max 30 frames)
  - Torso angles (deque, max 30 frames)
  - Head positions (deque, max 30 frames)
  - Velocity history (deque, max 10 frames)
- Purpose: Support temporal analysis over recent history

**Step 2: Extract Key Positions**
- Input: Current frame joints
- Extract:
  1. Hip height: (left_hip.y + right_hip.y) / 2
  2. Head position: nose.position (x, y)
  3. Torso angle: angle between shoulders and hips

**Step 3: Torso Angle Calculation**
```python
def calculate_torso_angle(joints):
    """
    Calculate angle of torso orientation (degrees from vertical).
    0° = perfectly vertical, 90° = horizontal
    """
    if not all joint in joints for joint in needed_joints:
        return None
    
    left_shoulder = joints['left_shoulder']['position']
    right_shoulder = joints['right_shoulder']['position']
    left_hip = joints['left_hip']['position']
    right_hip = joints['right_hip']['position']
    
    # Midpoints
    shoulder_mid = [(left_shoulder[0]+right_shoulder[0])/2, 
                    (left_shoulder[1]+right_shoulder[1])/2]
    hip_mid = [(left_hip[0]+right_hip[0])/2, 
               (left_hip[1]+right_hip[1])/2]
    
    # Vector from hip to shoulder
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    
    # Angle from vertical (0° = upright, 90° = horizontal)
    angle_from_vertical = abs(math.degrees(math.atan2(dx, dy)))
    
    return min(angle_from_vertical, 180 - angle_from_vertical)
```

**Step 4: Fall Indicator 1 - Hip Drop Detection**
- Input: Hip position history
- Condition: Last 5 frames
  ```
  hip_drop_speed = (hip_position[-1] - hip_position[-5]) / 5  [pixels/frame]
  hip_drop_distance = hip_position[-1] - min(hip_position[-5:])
  
  if hip_drop_speed > THRESHOLD (5.0 px/frame):
      fall_indicators["hip_drop"] = True
      fall_severity += 0.3
  if hip_drop_distance > THRESHOLD (50 px):
      fall_severity += 0.2
  ```
- Threshold values: 5.0 pixels/frame for speed, 50 pixels for distance
- Rationale: Rapid descent indicates loss of balance or falling

**Step 5: Fall Indicator 2 - Torso Orientation Change**
- Input: Torso angle history
- Condition: Last 5 frames
  ```
  angle_change = abs(torso_angle[-1] - torso_angle[-5])
  
  if angle_change > THRESHOLD (30 degrees):
      fall_indicators["torso_tilt"] = True
      fall_severity += 0.25
  ```
- Threshold: 30 degrees over 5 frames (~100ms at 30fps)
- Rationale: Sudden body rotation indicates dynamic fall event

**Step 6: Fall Indicator 3 - Head Downward Velocity**
- Input: Head vertical position history
- Condition: Last 3 frames
  ```
  head_velocity = (head_y[-1] - head_y[-3]) / 3  [pixels/frame]
  
  if head_velocity > THRESHOLD (4.0 px/frame, downward):
      fall_indicators["head_velocity"] = True
      fall_severity += 0.2
  ```
- Threshold: 4.0 pixels/frame downward motion
- Rationale: Head accelerating downward suggests active fall

**Step 7: Fall Indicator 4 - Ground Contact Detection**
- Input: Hip height, frame height
- Calculation:
  ```
  normalized_height = hip_y / frame_height
  
  if normalized_height > THRESHOLD (0.75 of frame):
      ground_contact_counter += 1
  else:
      ground_contact_counter = 0
  
  if ground_contact_counter >= DURATION_THRESHOLD (10 frames):
      fall_indicators["ground_contact"] = True
      fall_severity += 0.15
  ```
- Threshold: Hip at 75% of frame height (low position)
- Duration: 10 frames of sustained low position
- Rationale: Sustained low position indicates prone or supine state

**Step 8: Fall Indicator 5 - Sudden Deceleration**
- Input: Center of mass (COM) position history
- Calculation:
  ```
  if len(velocity_history) >= 5:
      recent_velocities = calculate_velocities(last 5 COM positions)
      
      deceleration = velocity[0] - velocity[-1]  [magnitude difference]
      
      if deceleration > THRESHOLD (8.0 px/frame):
          fall_indicators["sudden_stop"] = True
          fall_severity += 0.1
  ```
- Threshold: 8.0 pixels/frame deceleration
- Rationale: Rapid stop after fast movement suggests collision or fall

**Step 9: Condition Aggregation**
- Count how many indicators are True
- Requirement: >= fall_confirmation_conditions threshold (default 2)
- Formula: is_falling = (sum of True conditions) >= 2

**Step 10: Severity Normalization**
```python
fall_severity = min(fall_severity * 100, 100.0)
# Raw severity (0.0-1.0) scaled to 0-100 range
# Capped at 100
```

**Step 11: Fall Event Recording**
- If is_falling and not was_falling on previous frame:
  - Record fall event:
    ```json
    {
        "frame": frame_number,
        "severity": fall_severity,
        "indicators_triggered": ["hip_drop", "torso_tilt", ...],
        "timestamp": frame_number / fps
    }
    ```
  - Add to fall_events list
  - Update current_fall_state = True

**Step 12: Fall Event Termination**
- If was_falling on previous frame but not now:
  - Mark fall event as ended
  - Set current_fall_state = False
  - Allow new fall events to be detected

#### 3.4.3 Fall Detection Configuration

| Parameter | Value | Impact |
|-----------|-------|--------|
| hip_drop_speed_threshold | 5.0 px/frame | <3: too sensitive; >7: miss real falls |
| hip_drop_distance | 50 pixels | <30: false positives; >70: miss gradual falls |
| torso_angle_change | 30 degrees | <20: too many; >40: misses quick falls |
| head_velocity | 4.0 px/frame | <2: oversensitive; >6: misses rapid falls |
| ground_contact_height | 0.75 | Waist-level height threshold |
| ground_contact_duration | 10 frames | ~333ms, confirms sustained contact |
| sudden_stop_deceleration | 8.0 px/frame | <5: many false positives; >10: misses impacts |
| fall_confirmation_conditions | 2 | At least 2 indicators must trigger |

#### 3.4.4 Data Output from Stage 4

```json
{
  "frame_number": 250,
  "falling": true,
  "fall_severity": 72.5,
  "indicators": {
    "hip_drop": true,
    "torso_tilt": true,
    "head_velocity": false,
    "ground_contact": true,
    "sudden_stop": false
  },
  "hip_height": 380,
  "torso_angle": 45.2,
  "fall_events": [
    {
      "frame": 248,
      "severity": 72.5,
      "timestamp": 8.27
    }
  ]
}
```

---

### Stage 5: Motion Analysis

#### 3.5.1 Objective

Analyze raider motion patterns over time to identify kinematic anomalies that correlate with injury risk.

#### 3.5.2 Precise Implementation Steps

**Step 1: Joint Data Buffer Management**
- Input: Current frame joint positions and velocities
- Initialize buffers (if first frame):
  ```python
  for joint_name in joints.keys():
      joint_position_buffer[joint_name] = deque(maxlen=window_size)  # 10 frames default
      joint_velocity_buffer[joint_name] = deque(maxlen=window_size)
      joint_acceleration_buffer[joint_name] = deque(maxlen=window_size)
  ```
- Window size: 10 frames (~333ms at 30fps)

**Step 2: Update Joint Buffers**
- For each joint:
  ```python
  position = joints[joint_name]['position']
  
  joint_position_buffer[joint_name].append(position)
  ```

**Step 3: Velocity Calculation**
- Input: Joint positions from buffers
- Calculation:
  ```python
  if len(position_buffer) >= 2:
      prev_position = position_buffer[-2]
      curr_position = position_buffer[-1]
      
      velocity = curr_position - prev_position  [2D vector]
      velocity_buffer.append(velocity)
  ```
- Units: pixels/frame
- Output: 2D velocity vector (vx, vy)

**Step 4: Acceleration Calculation**
- Input: Joint velocities from buffers
- Calculation:
  ```python
  if len(velocity_buffer) >= 2:
      prev_velocity = velocity_buffer[-2]
      curr_velocity = velocity_buffer[-1]
      
      acceleration = curr_velocity - prev_velocity
      acceleration_buffer.append(acceleration)
  ```
- Units: pixels/frame²
- Output: 2D acceleration vector (ax, ay)

**Step 5: Jerk Calculation (Smoothness)**
- Input: Joint accelerations from buffers
- Calculation:
  ```python
  if len(acceleration_buffer) >= 2:
      accelerations = list(acceleration_buffer)
      
      for i in range(1, len(accelerations)):
          jerk = norm(accelerations[i] - accelerations[i-1])
          jerk_list.append(jerk)
      
      jerk_metrics = {
          "current": jerk_list[-1],
          "max": max(jerk_list),
          "avg": mean(jerk_list)
      }
  ```
- Units: pixels/frame³
- Interpretation: Lower jerk = smoother motion

**Step 6: Metric Calculation Per Joint**
- For each joint, calculate metrics over window:

| Metric | Calculation | Example values |
|--------|-----------|-----------------|
| Velocity magnitude | sqrt(vx²+vy²) | 0-20 px/frame |
| Max velocity | max(all velocities) | 2-30 px/frame |
| Avg velocity | mean(all velocities) | 1-15 px/frame |
| Accel magnitude | sqrt(ax²+ay²) | 0-5 px/frame² |
| Max accel | max(all accelerations) | 0.5-8 px/frame² |
| Jerk max | max(all jerks) | 0.1-2 px/frame³ |

**Step 7: Trajectory Smoothness**
```python
def calculate_smoothness(positions):
    """
    Calculate curvature-based smoothness metric.
    Lower values indicate smoother trajectories.
    """
    if len(positions) < 3:
        return 0.0
    
    smoothness_score = 0.0
    
    for i in range(1, len(positions)-1):
        p_prev = positions[i-1]
        p_curr = positions[i]
        p_next = positions[i+1]
        
        # Vectors
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        # Angle between consecutive segments
        cos_angle = dot(v1, v2) / (norm(v1)*norm(v2) + 1e-6)
        angle = arccos(clip(cos_angle, -1, 1))
        
        smoothness_score += angle
    
    return smoothness_score / (len(positions) - 2)
```
- Output: Smoothness score (0-π, lower is smoother)

**Step 8: Left-Right Asymmetry**
- Input: Metrics for bilateral joints (left/right pairs)
- Joint pairs:
  - Shoulders: left_shoulder vs right_shoulder
  - Elbows: left_elbow vs right_elbow
  - Hips: left_hip vs right_hip
  - Knees: left_knee vs right_knee
  - Ankles: left_ankle vs right_ankle
- Calculation:
  ```python
  asymmetry_score = {}
  
  for pair_name, (left_joint, right_joint) in joint_pairs.items():
      left_vel_mag = norm(left_metrics['velocity']['avg'])
      right_vel_mag = norm(right_metrics['velocity']['avg'])
      
      if (left_vel_mag + right_vel_mag) > 0:
          asymmetry = abs(left_vel_mag - right_vel_mag) / (left_vel_mag + right_vel_mag)
          asymmetry_score[pair_name] = asymmetry
  ```
- Output: Dict[pair_name -> asymmetry_score], range [0, 1]
- Interpretation: > 0.3 indicates significant asymmetry

**Step 9: Motion Abnormality Score (Rule-Based)**
```python
def calculate_abnormality_score(metrics):
    """
    Aggregate all motion metrics into single abnormality score (0-100).
    """
    abnormality = 0.0
    
    # High velocity -> decreased control
    avg_all_velocities = mean all velocity.avg values
    if avg_all_velocities > 15:  # pixels/frame
        abnormality += (avg_all_velocities - 15) * 2
    
    # High acceleration -> explosive/unstable
    avg_all_accels = mean all acceleration.avg values
    if avg_all_accels > 3:  # px/frame²
        abnormality += (avg_all_accels - 3) * 5
    
    # High jerk -> jerky/uncontrolled
    avg_all_jerks = mean all jerk.avg values
    if avg_all_jerks > 0.5:  # px/frame³
        abnormality += (avg_all_jerks - 0.5) * 10
    
    # High asymmetry -> imbalance/compensation
    avg_asymmetry = mean all asymmetry_scores
    if avg_asymmetry > 0.3:
        abnormality += (avg_asymmetry - 0.3) * 50
    
    # Low smoothness -> choppy
    avg_smoothness = mean all smoothness scores
    if avg_smoothness > 1.0:  # radians
        abnormality += (avg_smoothness - 1.0) * 20
    
    return clip(abnormality, 0, 100)
```

**Step 10: LSTM Temporal Analysis (Optional)**
- Input: Motion metrics over longer window (90 frames)
- If LSTM model available:
  - Load pretrained model: `motion_lstm_model.pth`
  - Extract features: velocity, acceleration, jerk, asymmetry
  - LSTM forward pass: outputs temporal abnormality score
  - Combine with rule-based score: final_score = 0.6*lstm + 0.4*rules
- If LSTM unavailable:
  - Use rule-based score only

**Step 11: Abnormality Score Temporal Smoothing**
```python
# Maintain history of abnormality scores
abnormality_scores.append(current_abnormality)

if len(abnormality_scores) > smoothing_window:
    smoothed_score = median(abnormality_scores[-smoothing_window:])
else:
    smoothed_score = mean(abnormality_scores)
```
- Smoothing window: 5 frames
- Purpose: Reduce false positives from single-frame noise

#### 3.5.3 Motion Analysis Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| window_size | 10 frames | 333ms: captures short-term dynamics |
| velocity_threshold | 15 px/frame | ~450 px/s at 30fps, abnormally fast |
| acceleration_threshold | 3 px/frame² | Rapid direction/speed changes |
| jerk_threshold | 0.5 px/frame³ | Very jerky/uncontrolled |
| asymmetry_threshold | 0.3 | 30% difference between sides |
| smoothness_threshold | 1.0 radian | High curvature = awkward |
| smoothing_window | 5 frames | Reduce transient noise |

#### 3.5.4 Data Output from Stage 5

```json
{
  "frame_number": 300,
  "motion_metrics": {
    "left_shoulder": {
      "velocity": {
        "current": 12.5,
        "max": 22.3,
        "avg": 8.6,
        "std": 3.2
      },
      "acceleration": {
        "current": 1.2,
        "max": 4.5,
        "avg": 0.8
      },
      "jerk": {
        "current": 0.3,
        "max": 0.9,
        "avg": 0.2
      },
      "smoothness": 0.65
    },
    # ... 16 more joints
  },
  "asymmetry": {
    "shoulders": 0.15,
    "elbows": 0.28,
    "hips": 0.12,
    "knees": 0.32,
    "ankles": 0.18
  },
  "motion_abnormality_score": 35.2,
  "lstm_score": 38.5,
  "final_abnormality": 36.3
}
```

---

### Stage 6: Impact Detection

#### 3.6.1 Objective

Detect collisions between raider and defenders, analyze defender approach angles, and quantify impact severity.

#### 3.6.2 Precise Implementation Steps

**Step 1: Defender Position History Tracking**
- Input: All players except raider
- For each defender track_id:
  - Initialize history: deque[Dict] (position, bbox, frame)
  - Max history length: 30 frames
- On each frame:
  - Append: {"center": (x, y), "bbox": [x1, y1, x2, y2], "frame": frame_num}
  - Automatically evict oldest entries when full

**Step 2: Collision Detection (Proximity-Based)**
- Input: Raider position, defender positions, frame number
- Collision radius: 100 pixels (configurable)
- Algorithm:
  ```python
  raider_x, raider_y = raider['center']
  colliding_defenders = []
  
  for defender in defenders:
      def_x, def_y = defender['center']
      
      distance = sqrt((raider_x - def_x)² + (raider_y - def_y)²)
      
      if distance < COLLISION_RADIUS (100 px):
          colliding_defenders.append(defender['track_id'])
          impact_score += 0.3  # Increment collision impact
  ```
- Output: List of colliding defender IDs
- Impact score increment: 0.3 per collision, capped at 1.0

**Step 3: Defender Velocity Calculation**
- Input: Defender position history (last 30 frames)
- For each defender:
  ```python
  if len(history) >= 5:
      recent_positions = history[-5:]  # Last ~167ms
      
      velocities = []
      for i in range(1, len(recent_positions)):
          p1 = recent_positions[i-1]['center']
          p2 = recent_positions[i]['center']
          
          vel = (p2[0] - p1[0], p2[1] - p1[1])
          velocities.append(vel)
      
      avg_velocity = mean(velocities)  # 2D vector
  ```
- Output: Average velocity vector [vx, vy] in pixels/frame
- Usage: For impact angle calculation

**Step 4: Approach Angle Calculation**
- Input: Defender position, defender velocity, raider position
- Algorithm:
  ```python
  def_pos = array(defender_position)
  raider_pos = array(raider_position)
  def_vel = array(defender_velocity)
  
  # Vector from defender to raider
  to_raider = raider_pos - def_pos
  
  # Normalize both vectors
  def_vel_norm = def_vel / (norm(def_vel) + 1e-6)
  to_raider_norm = to_raider / (norm(to_raider) + 1e-6)
  
  # Angle between velocity and direction to raider
  cos_angle = dot(def_vel_norm, to_raider_norm)
  cos_angle = clip(cos_angle, -1, 1)
  
  angle = degrees(arccos(cos_angle))
  ```
- Output: Angle in degrees (0-180)
  - 0° = directly toward raider (head-on collision)
  - 90° = perpendicular approach
  - 180° = moving away from raider
- Interpretation: Lower angle = more aggressive approach

**Step 5: Impact Severity Calculation (Per Defender)**
```python
def calculate_impact_severity(defender_metrics):
    """
    Combine distance, approach angle, and velocity for impact score.
    """
    distance = defender_metrics['distance']
    approach_angle = defender_metrics['approach_angle']
    velocity_magnitude = defender_metrics['velocity_magnitude']
    
    severity = 0.0
    
    # Component 1: Close proximity (0-50 points)
    if distance < 50:
        proximity_score = (1 - distance/50) * 50
    else:
        proximity_score = 0.0
    
    # Component 2: Head-on approach (0-30 points)
    if approach_angle < 45:  # Directly toward raider
        approach_score = (1 - approach_angle/45) * 30
    else:
        approach_score = 0.0
    
    # Component 3: High velocity (0-20 points)
    if velocity_magnitude > 5:  # px/frame
        velocity_score = min((velocity_magnitude - 5) / 10, 1.0) * 20
    else:
        velocity_score = 0.0
    
    severity = proximity_score + approach_score + velocity_score
    
    return clip(severity, 0, 100)
```

**Step 6: Multi-Defender Impact Aggregation**
```python
def aggregate_defender_impacts(all_defender_impacts):
    """
    Combine multiple defender impacts into single impact severity.
    """
    if not all_defender_impacts:
        return 0.0
    
    # Sort by severity
    sorted_impacts = sorted(all_defender_impacts, reverse=True)
    
    # Take top 3 defenders (if available)
    top_impacts = sorted_impacts[:3]
    
    # Weighted average: closest defender weighted more
    total_severity = 0.0
    total_weight = 0.0
    
    for idx, impact in enumerate(top_impacts):
        weight = 1.0 / (idx + 1)  # 1st defender: weight 1, 2nd: 0.5, 3rd: 0.33
        total_severity += impact * weight
        total_weight += weight
    
    final_impact_severity = total_severity / total_weight
    
    return clip(final_impact_severity, 0, 100)
```

**Step 7: Impact Event Recording**
- If impact_severity > 0:
  ```json
  {
      "frame": frame_number,
      "raider_id": raider_id,
      "colliding_defenders": [1, 3, 5],
      "impact_severity": 68.5,
      "defender_distances": [45, 85, 120],
      "defender_approach_angles": [15, 60, 90],
      "timestamp": frame_number / fps
  }
  ```
- Store in impact_events list
- Accumulate for event timeline

**Step 8: Temporal Impact Smoothing**
```python
# Maintain impact score history (last 10 frames)
impact_history = deque(maxlen=10)
impact_history.append(current_impact_severity)

# Smooth with moving average
smoothed_impact_severity = mean(list(impact_history))
```
- Purpose: Reduce jitter in severity calculations
- Smoothing window: 10 frames (~333ms)

#### 3.6.3 Impact Detection Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| collision_radius | 100 pixels | ~100px for avg player size at 30fps |
| approach_radius | 300 pixels | Look ahead for incoming defenders |
| defender_velocity_window | 5 frames | 167ms for reliable velocity estimation |
| proximity_weight | 50 points | Closest defenders most dangerous |
| approach_angle_weight | 30 points | Head-on more impactful than glancing |
| velocity_weight | 20 points | Higher speed increases impact |
| top_defenders_count | 3 | Consider worst 3 threats |
| impact_smoothing_window | 10 frames | Smooth over ~333ms |

#### 3.6.4 Data Output from Stage 6

```json
{
  "frame_number": 400,
  "raider_id": 7,
  "impact_detected": true,
  "colliding_defenders": [2, 4],
  "impact_severity": 65.3,
  "defender_impacts": [
    {
      "defender_id": 2,
      "distance": 45.2,
      "approach_angle": 22.5,
      "velocity": 7.8,
      "impact_severity": 78.5
    },
    {
      "defender_id": 4,
      "distance": 98.5,
      "approach_angle": 75.0,
      "velocity": 3.2,
      "impact_severity": 18.2
    }
  ],
  "overall_impact_severity": 65.3
}
```

---

### Stage 7: Risk Fusion and Final Score Calculation

#### 3.7.1 Objective

Synthesize all injury risk components (falls, impacts, motion anomalies, injury history) into a unified final risk score with explainable breakdown.

#### 3.7.2 Precise Implementation Steps

**Step 1: Load Injury History Database**
- Input: CSV file at `data/injury_history.csv`
- Expected columns:
  ```
  player_id, injury_count, severity_avg, risk_modifier, notes
  ```
- Processing:
  - Read with pandas, handling multiple encodings (UTF-8, Latin-1, ISO-8859-1)
  - Create player_id indexed lookup dictionary
  - If file missing or unreadable, create empty DataFrame
- Output: injury_history_df (pandas DataFrame)

**Step 2: Injury History Risk Modifier Lookup**
```python
def get_injury_history_modifier(player_id: int) -> float:
    """
    Lookup injury history risk modifier for player.
    Returns 0-100 scale.
    """
    if injury_history_df is None or len(injury_history_df) == 0:
        return 0.0
    
    player_record = injury_history_df[injury_history_df['player_id'] == str(player_id)]
    
    if len(player_record) == 0:
        return 0.0  # No history for this player
    
    # Use explicit risk_modifier if available
    if 'risk_modifier' in player_record.columns:
        modifier = player_record.iloc[0]['risk_modifier']
        return float(modifier) if not pd.isna(modifier) else 0.0
    
    # Calculate from injury count and severity
    injury_count = player_record.iloc[0].get('injury_count', 0)
    severity_avg = player_record.iloc[0].get('severity_avg', 0)
    
    modifier = min(injury_count * 10 + severity_avg * 0.5, 100.0)
    return modifier
```
- Output: Modifier value (0-100)

**Step 3: Collect Component Risk Scores**
- Input: Outputs from Stages 4, 5, 6
  ```python
  fall_severity = stage4_output['fall_severity']  # 0-100
  impact_severity = stage6_output['impact_severity']  # 0-100
  motion_abnormality = stage5_output['motion_abnormality_score']  # 0-100
  injury_history_score = get_injury_history_modifier(raider_id)  # 0-100
  ```
- All normalized to 0-100 range

**Step 4: Apply Weighted Fusion**
```python
def calculate_risk_score(fall, impact, motion, history):
    """
    Weighted fusion of risk components.
    """
    weights = {
        "fall_severity": 0.4,        # 40% weight - most predictive
        "impact_severity": 0.3,      # 30% weight - significant
        "motion_abnormality": 0.2,   # 20% weight - moderate
        "injury_history": 0.1        # 10% weight - baseline risk
    }
    
    raw_score = (
        weights["fall_severity"] * fall +
        weights["impact_severity"] * impact +
        weights["motion_abnormality"] * motion +
        weights["injury_history"] * history
    )
    
    # Normalize to 0-100 (already should be due to weighting)
    risk_score = clip(raw_score, 0, 100)
    
    return risk_score
```
- Formula: risk = 0.4*fall + 0.3*impact + 0.2*motion + 0.1*history
- Output: Single score (0-100)

**Step 5: Temporal Smoothing** (Optional, if enabled)
```python
# Maintain history of risk scores
risk_score_history.append(current_risk_score)

if len(risk_score_history) >= smoothing_window:
    smoothed_score = mean(risk_score_history[-smoothing_window:])
else:
    smoothed_score = mean(risk_score_history)

# Use smoothed score as final
final_risk_score = smoothed_score
```
- Smoothing window: 5 frames
- Purpose: Reduce transient spikes, emphasize sustained risk

**Step 6: Risk Level Classification**
```python
def get_risk_level(score: float) -> str:
    """
    Map numerical score to categorical risk level.
    """
    if score <= 30:
        return "LOW"
    elif score <= 70:
        return "MEDIUM"
    else:
        return "HIGH"
```

| Risk Level | Score Range | Interpretation | Action |
|-----------|-------------|-----------------|--------|
| LOW | 0-30 | Safe, normal play | Continue monitoring |
| MEDIUM | 31-70 | Elevated risk, needs attention | Alert personnel |
| HIGH | 71-100 | Acute danger | Immediate intervention |

**Step 7: Generate Risk Breakdown**
```python
risk_breakdown = {
    "fall_component": fall_severity,
    "impact_component": impact_severity,
    "motion_component": motion_abnormality,
    "history_component": injury_history_score,
    "weights": {
        "fall": 0.4,
        "impact": 0.3,
        "motion": 0.2,
        "history": 0.1
    },
    "contributions": {
        "fall": 0.4 * fall_severity,
        "impact": 0.3 * impact_severity,
        "motion": 0.2 * motion_abnormality,
        "history": 0.1 * injury_history_score
    }
}
```

**Step 8: Store Risk Event**
```python
risk_event = {
    "frame": frame_number,
    "timestamp": frame_number / fps,
    "raider_id": raider_id,
    "risk_score": final_risk_score,
    "risk_level": risk_level,
    "breakdown": risk_breakdown,
    "is_peak": final_risk_score > previous_risk_score  # Peak detection
}

risk_timeline.append(risk_event)
```

**Step 9: Peak Risk Detection**
```python
# Detect local maxima (peaks) in risk timeline
def detect_risk_peaks(risk_timeline, window=5):
    """
    Identify frames where risk score is locally maximum.
    """
    peaks = []
    
    for i in range(window, len(risk_timeline) - window):
        current_score = risk_timeline[i]['risk_score']
        
        # Check if local maximum
        is_peak = all(
            current_score > risk_timeline[j]['risk_score']
            for j in range(i-window, i+window+1)
            if j != i
        )
        
        if is_peak and current_score > 50:  # Only peaks > 50
            peaks.append({
                "frame": risk_timeline[i]['frame'],
                "score": current_score,
                "level": risk_timeline[i]['risk_level']
            })
    
    return peaks
```

**Step 10: Risk Score History Management**
```python
# Temporal statistics
risk_stats = {
    "mean": mean(all_risk_scores),
    "max": max(all_risk_scores),
    "min": min(all_risk_scores),
    "std": std(all_risk_scores),
    "peak_count": len(peaks),
    "high_risk_duration": sum frames where score > 71,
    "elevated_risk_duration": sum frames where 31 < score <= 70
}
```

**Step 11: Final Risk Report Generation**
```python
final_report = {
    "raider_id": raider_id,
    "video_duration_seconds": total_frames / fps,
    "overall_risk_level": determine_overall_level(risk_stats),
    "summary_statistics": risk_stats,
    "peak_risk_events": peaks,
    "risk_timeline_points": 30,  # Sample 30 points for visualization
    "recommendations": generate_medical_recommendations(risk_stats)
}
```

#### 3.7.3 Risk Fusion Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| fall_weight | 0.40 | Falls are strongest injury predictor |
| impact_weight | 0.30 | High-speed collisions very dangerous |
| motion_weight | 0.20 | Abnormal movement less direct |
| history_weight | 0.10 | Previous injury slightly increases risk |
| smoothing_enabled | True | Reduce transient noise |
| smoothing_window | 5 frames | 167ms smoothing window |
| peak_threshold | 50 | Only record peaks above LOW risk |
| high_risk_threshold | 71 | Score > 71 = HIGH |
| medium_risk_threshold | 31 | Score 31-70 = MEDIUM |

#### 3.7.4 Data Output from Stage 7

```json
{
  "frame_number": 500,
  "timestamp": 16.67,
  "raider_id": 7,
  "final_risk_score": 68.2,
  "risk_level": "MEDIUM",
  "component_scores": {
    "fall_severity": 72.5,
    "impact_severity": 65.3,
    "motion_abnormality": 52.1,
    "injury_history": 28.0
  },
  "contributions": {
    "fall_component": 29.0,
    "impact_component": 19.59,
    "motion_component": 10.42,
    "history_component": 2.8
  },
  "breakdown": {
    "weights": {
      "fall": 0.4,
      "impact": 0.3,
      "motion": 0.2,
      "history": 0.1
    },
    "is_peak_risk": true,
    "peak_rank": 3
  },
  "temporal_stats": {
    "mean_risk": 42.3,
    "max_risk": 78.5,
    "min_risk": 12.1,
    "std_risk": 18.7,
    "peak_count": 5,
    "high_risk_frames": 45,
    "elevated_risk_frames": 156
  }
}
```

---

## 4. Data Structure Specifications

### 4.1 Core Data Types

#### Player Detection Structure
```python
class PlayerDetection:
    track_id: int                    # Unique identifier (0-N)
    bbox: List[float]                # [x1, y1, x2, y2] in pixels
    center: Tuple[int, int]          # (cx, cy) in pixels
    frame_num: int                   # Frame number (0-based)
    confidence: float                # Detection confidence (0-1)
```

#### Joint Structure
```python
class Joint:
    position: List[float]            # [x, y] in pixels
    velocity: List[float]            # [vx, vy] pixels/frame
    visibility: float                # Landmark visibility (0-1)
    history: List[List[float]]       # Last N positions
```

#### Risk Event Structure
```python
class RiskEvent:
    frame: int                       # Frame number
    timestamp: float                 # Time in seconds
    raider_id: int                   # Which raider (track ID)
    risk_score: float                # Final risk (0-100)
    risk_level: str                  # "LOW" | "MEDIUM" | "HIGH"
    breakdown: Dict[str, float]      # Component contributions
    metadata: Dict[str, Any]         # Additional context
```

### 4.2 JSON Output Schema

#### Pipeline Summary Output
```json
{
  "metadata": {
    "video_path": "string",
    "video_duration": "float (seconds)",
    "fps": "float",
    "total_frames": "int",
    "processing_duration": "float (seconds)",
    "timestamp": "ISO 8601 datetime"
  },
  "detection_summary": {
    "total_players": "int",
    "raider_id": "int or null",
    "total_frames_with_raider": "int"
  },
  "risk_summary": {
    "mean_risk_score": "float",
    "max_risk_score": "float",
    "min_risk_score": "float",
    "std_risk_score": "float",
    "peak_risk_frames": "int",
    "high_risk_frames": "int",
    "elevated_risk_frames": "int"
  },
  "event_summary": {
    "total_falls": "int",
    "total_collisions": "int",
    "total_high_risk_events": "int"
  },
  "stage_completion": {
    "stage_0": "bool",
    "stage_1": "bool",
    "stage_2": "bool",
    "stage_3": "bool",
    "stage_4": "bool",
    "stage_5": "bool",
    "stage_6": "bool",
    "stage_7": "bool"
  }
}
```

---

## 5. Integration Points and Communication

### 5.1 Inter-Stage Data Flow

```
┌─────────────────────────────────────────┐
│  Stage 0: Court Line Detection          │
│  Output: court_lines.json               │
└──────────────┬──────────────────────────┘
               │
               v
┌─────────────────────────────────────────┐
│  Stage 1: Player Detection & Tracking   │
│  Input: detection_line_x from Stage 0   │
│  Output: players_list                   │
└──────────────┬──────────────────────────┘
               │
               v
┌─────────────────────────────────────────┐
│  Stage 2: Raider Identification         │
│  Input: players_list, detection_line_x  │
│  Output: raider_id                      │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴────────┐
        v               v
    ┌────────┐    ┌──────────┐
    │Stage 3 │    │Stage 6   │
    │  Pose  │    │ Impact   │
    └───┬────┘    └────┬─────┘
        │              │
        v              │
    ┌────────┐         │
    │Stage 4 │         │
    │  Fall  │         │
    └───┬────┘         │
        │              │
        v              │
    ┌────────┐         │
    │Stage 5 │         │
    │ Motion │         │
    └───┬────┴────┬────┘
        │         │
        v         v
    ┌─────────────────────────┐
    │  Stage 7: Risk Fusion   │
    │  Inputs: All scores     │
    │  Output: final_risk     │
    └──────────┬──────────────┘
               │
               v
    ┌─────────────────────────┐
    │  Annotation & Export    │
    │  Output: final_output.mp4
    └─────────────────────────┘
```

### 5.2 Shared State Management

```
PipelineSession
├── PipelineStatus: Stage completion tracking
├── MetricsLogger: Frame-by-frame metrics (CSV export)
├── EventLogger: Real-time event streaming
├── ValueLogger: Intermediate calculation values
├── RaiderStatusTracker: Raider ID persistence
└── CollisionHistory: Accumulated collision events

All maintain JSON/CSV files for Streamlit dashboard
```

### 5.3 Real-Time Interface (Streamlit Dashboard)

```
Live Dashboard Updates:
├── Status Cards: Stage processing progress
├── Event Log: Last 4 detected events
├── Risk Meter: Current risk score gauge
├── Video Player: Processing video preview
├── Event Timeline: Interactive risk history
└── Metrics Tables: Detailed statistics

File-based communication:
├── live_events.json (updated every frame)
├── pipeline_status.json (updated per stage)
└── pipeline_summary.json (final report)
```

---

## 6. Performance Metrics and Outputs

### 6.1 Output Files Generated

After each video processing, the system creates:

```
outputs/
├── session_YYYYMMDD_HHMMSS/
│   ├── final_output.mp4              # Annotated video (1080p, H.264)
│   ├── pipeline_summary.json         # Complete analysis report
│   ├── metrics.json                  # Frame-by-frame metrics
│   ├── events_timeline.json          # Temporal event sequence
│   ├── collision_data.json           # Collision events list
│   ├── pipeline_status.json          # Stage completion status
│   ├── calculation_values/
│   │   ├── stage0_values.csv         # Court lines metrics
│   │   ├── stage1_values.csv         # Detection metrics
│   │   ├── stage2_values.csv         # Raider metrics
│   │   ├── stage3_values.csv         # Pose joint data
│   │   ├── stage4_values.csv         # Fall detection
│   │   ├── stage5_values.csv         # Motion analysis
│   │   ├── stage6_values.csv         # Impact detection
│   │   └── stage7_values.csv         # Final risk scores
│   ├── logs/
│   │   ├── pipeline_YYYYMMDD_HHMMSS.log  # Complete debug log
│   │   └── logs_index.json                 # Log file index
│   ├── stage0_court_lines/
│   │   └── court_lines.json
│   ├── stage1_detection/
│   │   └── detections_frame_*.json
│   ├── stage2_raider/
│   │   └── raider_frame_*.json
│   ├── stage3_pose/
│   │   └── pose_frame_*.json
│   ├── stage4_falls/
│   │   └── fall_events.json
│   ├── stage5_motion/
│   │   └── motion_frame_*.json
│   ├── stage6_impact/
│   │   └── impact_frame_*.json
│   └── stage7_final/
│       └── risk_frame_*.json
```

### 6.2 Performance Statistics

| Metric | Expected Value | Hardware |
|--------|----------------|----------|
| Frame processing rate | 25-30 FPS | GPU (CUDA) |
| Peak memory usage | 4-6 GB | Training dataset |
| Total processing time (1 hour video) | ~120-150 seconds | GPU |
| Model inference time per frame | 30-40ms | GPU |
| Post-processing per frame | 5-10ms | CPU |

### 6.3 Accuracy Metrics

| Component | Metric | Target |
|-----------|--------|--------|
| Player detection (Stage 1) | mAP @ IoU=0.5 | > 0.95 |
| Raider identification (Stage 2) | Accuracy | > 0.98 |
| Pose estimation (Stage 3) | PCK @ 0.2 | > 0.90 |
| Fall detection (Stage 4) | Sensitivity | > 0.85 |
| Impact detection (Stage 6) | Precision | > 0.80 |
| Risk fusion (Stage 7) | Calibration error | < 10 points |

---

## 7. Conclusion

This comprehensive Kabaddi Injury Prediction System represents a sophisticated multi-stage analytical pipeline designed to extract actionable injury risk insights from raw video footage. Each stage operates independently yet contributes meaningfully to the final risk assessment through carefully engineered feature extraction, sophisticated signal processing, and principled multi-factor fusion.

The system's architecture emphasizes:
- **Modularity**: Each stage can be tested, debugged, and improved independently
- **Transparency**: Clear data flow and explicit parameters enable interpretation
- **Robustness**: Temporal smoothing, multiple indicators, and history-based methods reduce false positives
- **Scalability**: GPU acceleration and efficient algorithms support real-time processing

The complete data transformation from raw frames to final risk scores is fully traceable, logged, and exported, enabling continuous improvement through data-driven analysis and validation.

---

**Document Version:** 1.0
**Last Updated:** February 27, 2026
**Author:** Technical Documentation Team
**Classification:** Internal - Technical Reference

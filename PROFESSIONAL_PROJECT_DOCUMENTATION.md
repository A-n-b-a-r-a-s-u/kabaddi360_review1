# Kabaddi Injury Prediction System
## Professional Project Documentation

**Document Version:** 1.0  
**Date:** March 2026  
**Classification:** Technical Documentation  
**Status:** Final Review

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Module Description](#module-description)

---

## System Architecture

### 1.0 Executive Overview

The Kabaddi Injury Prediction System is a comprehensive, modular, video-based analysis pipeline designed to assess injury risk in Kabaddi match participants in real-time. The system employs a seven-stage sequential processing architecture combined with parallel computation streams to deliver accurate, explainable injury risk predictions without requiring wearable sensors or inertial measurement units (IMUs).

**Key Architectural Principles:**
- **Modular Design:** Independent, loosely-coupled stages with standardized interfaces
- **Video-Only Input:** No external sensors required; analysis based purely on video frames
- **Explainable AI:** Each risk component is individually traceable and interpretable
- **Hybrid Approach:** Combines rule-based biomechanical analysis with deep learning models
- **Real-Time Capable:** Optimized for processing video streams with minimal latency
- **Scalable Infrastructure:** Supports batch processing, streaming input, and concurrent clients

### 1.1 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VIDEO INPUT STREAM                               │
│              (Kabaddi Match Video File or Stream)                   │
└────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   STAGE 0: COURT DETECTION                          │
│           (White Line Detection via HSV + Morphological Ops)        │
├─────────────────────────────────────────────────────────────────────┤
│ Output: Court Coordinates | Processed Frame Metadata                │
└────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│             STAGE 1: PLAYER DETECTION & TRACKING                    │
│           (YOLOv8 + SORT Tracker for Consistent IDs)                │
├─────────────────────────────────────────────────────────────────────┤
│ Output: Bounding Boxes | Track IDs | Confidence Scores              │
└────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 2: RAIDER IDENTIFICATION                         │
│    (Multi-Cue Analysis: Court Logic + Motion + Temporal Consistency)│
├─────────────────────────────────────────────────────────────────────┤
│ Output: Raider Track ID | Identification Confidence                 │
└────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ RAIDER TRACK ID │
                     └────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
    ┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
    │   STAGE 3:      │ │  STAGE 4:    │ │  STAGE 6:        │
    │ POSE ESTIM.     │ │ FALL DETECT. │ │ IMPACT DETECT.   │
    │ (MediaPipe)     │ │ (Biom. Rules)│ │ (Defender Prox)  │
    │                 │ │              │ │                  │
    │ Joint Positions │ │ Fall Score   │ │ Collision Energy │
    │ Confidence      │ │ Indicators   │ │ Defender Vel.    │
    │                 │ │ Severity     │ │ Proximity Risk   │
    └────────┬────────┘ └──────┬───────┘ └────────┬─────────┘
              │                 │                  │
              └─────────────────┼──────────────────┘
                                │
                                ▼
                        ┌───────────────────┐
                        │   STAGE 5:        │
                        │ MOTION ANALYSIS   │
                        │  (Temporal LSTM   │
                        │  + Rule-Based)    │
                        │                   │
                        │ Motion Abnormal.  │
                        │ Joint Stress      │
                        │ Velocity/Accel.   │
                        │ Jerk Analysis     │
                        └─────────┬─────────┘
                                  │
            ┌─────────────────────┴────────────────────────┐
            │                                               │
            ▼                                               ▼
    ┌──────────────────┐                          ┌──────────────────┐
    │ STAGE 7.1:       │                          │ STAGE 7.2:       │
    │ RISK FUSION      │                          │ TEMPORAL SMOOTH. │
    │ (Multi-Factor)   │                          │ (History Buffer) │
    │                  │                          │                  │
    │ - Fall Score     │                          │ Smoothing Window │
    │ - Motion Score   │ ────────────────────────▶ │ Rolling Average  │
    │ - Impact Score   │                          │ Trend Analysis   │
    │ - History Factor │                          │                  │
    └─────────┬────────┘                          └──────────────────┘
              │
              ▼
    ┌──────────────────────────┐
    │  FINAL INJURY RISK SCORE │
    │ (0-100 Scale)            │
    │ Risk Level Classification │
    │ Component Breakdown       │
    └─────────┬────────────────┘
              │
      ┌───────┴───────┐
      │               │
      ▼               ▼
┌──────────────┐ ┌─────────────────────────┐
│ ANNOTATED    │ │   REAL-TIME SERVER      │
│ VIDEO OUTPUT │ │   - WebSocket Events    │
│              │ │   - HTTP REST Endpoints │
│ - Falls      │ │   - Event Broadcasting  │
│ - Collisions │ │   - Client Monitoring   │
│ - Risk Zones │ │                         │
│ - Scores     │ │                         │
└──────────────┘ └─────────────────────────┘
      │                     │
      └─────────┬───────────┘
                │
                ▼
    ┌────────────────────────────┐
    │  FINAL OUTPUT DELIVERABLES │
    │                            │
    │ - Annotated Video File     │
    │ - Event Timeline (JSON)    │
    │ - Risk Assessment Report   │
    │ - Collision Data           │
    │ - Metrics Summary          │
    │ - Session Logs             │
    └────────────────────────────┘
```

### 1.2 Data Flow Architecture

#### 1.2.1 Input-Processing-Output Pipeline

```
INPUT LAYER
├── Video File (MP4, AVI, MOV, etc.)
├── Frame Parameters (FPS, Resolution)
└── Processing Configuration (thresholds, weights)
           │
           ▼
PROCESSING LAYER
├── Frame Extraction & Preprocessing
├── Court/Contextual Analysis
├── Entity Detection & Tracking
├── Biomechanical Analysis
├── Temporal Pattern Recognition
└── Multi-Cue Fusion & Risk Calculation
           │
           ▼
OUTPUT LAYER
├── Annotated Video Stream
├── Event Logs (JSON/CSV)
├── Risk Timeline Data
├── Performance Metrics
└── Session Documentation
```

#### 1.2.2 Information Flow Between Stages

| Stage | Input Data | Processing | Output Data |
|-------|-----------|-----------|-----------|
| Stage 0 | Raw Frame | HSV Color Filtering + Morphology | Court Coordinates |
| Stage 1 | Raw Frame | YOLOv8 Detection + SORT Tracking | Player Bounding Boxes + IDs |
| Stage 2 | Player Tracks | Court Logic + Motion Signature Analysis | Raider ID + Confidence |
| Stage 3 | Frame + Raider Bbox | MediaPipe Pose Estimation | 33-Joint Skeleton (Raider Only) |
| Stage 4 | Skeleton Data + Frame | Biomechanical Rule Evaluation | Fall Indicators + Severity |
| Stage 5 | Skeleton Temporal Sequence | LSTM Analysis + Motion Metrics | Motion Abnormality Score |
| Stage 6 | All Player Tracks + Raider ID | Proximity + Velocity Analysis | Collision Energy + Impact Score |
| Stage 7 | All Risk Components | Weighted Fusion | Final Injury Risk Score (0-100) |

### 1.3 System Components Architecture

#### 1.3.1 Core Processing Engine

```
KabaddiInjuryPipeline (main.py)
├── Stage Orchestrator
│   ├── Sequential Stage Execution
│   ├── Frame-by-Frame Processing
│   ├── Error Handling & Recovery
│   └── Progress Tracking
│
├── Resource Management
│   ├── CUDA/CPU Device Selection
│   ├── Memory Optimization
│   └── Model Loading
│
├── Output Management
│   ├── Video Writing
│   ├── Metrics Logging
│   ├── Event Tracking
│   └── Session Management
│
└── Integration Layer
    ├── Server Communication
    ├── WebSocket Broadcasting
    └── Event Submission
```

#### 1.3.2 Model Components

```
DETECTION SUBSYSTEM
├── CourtLineDetector (Stage 0)
│   └── Technology: HSV + Morphological Operations
│
├── PlayerDetector (Stage 1)
│   └── Technology: YOLOv8 + SORT Tracking
│
└── RaiderIdentifier (Stage 2)
    └── Technology: Multi-Cue Fusion (Logic + Motion + Temporal)

ANALYSIS SUBSYSTEM
├── PoseEstimator (Stage 3)
│   └── Technology: MediaPipe (33-Joint Model)
│
├── FallDetector (Stage 4)
│   └── Technology: Rule-Based Biomechanical Analysis
│
└── MotionAnalyzer (Stage 5)
    └── Technology: Hybrid (LSTM + Rule-Based)

FUSION SUBSYSTEM
├── ImpactDetector (Stage 6)
│   └── Technology: Proximity-Based Physics Model
│
└── RiskFusionEngine (Stage 7)
    └── Technology: Weighted Multi-Factor Fusion
```

#### 1.3.3 Infrastructure Components

```
SERVER INFRASTRUCTURE (FastAPI)
├── HTTP REST API
│   ├── /event/raider-identified
│   ├── /event/injury-risk
│   ├── /event/collision
│   ├── /event/fall
│   └── /health
│
├── WebSocket Server (/ws)
│   ├── Real-Time Event Broadcasting
│   ├── Client Connection Management
│   └── Async Event Queue
│
└── Middleware
    ├── CORS Handling
    ├── Request Validation
    └── Error Handling

VISUALIZATION LAYER
├── Streamlit Web Dashboard
│   ├── Video Upload & Management
│   ├── Real-Time Processing Display
│   ├── Results Visualization
│   └── Historical Analysis
│
└── Video Annotation Engine
    ├── Bounding Box Rendering
    ├── Skeleton Overlay
    ├── Risk Score Display
    └── Event Annotation
```

### 1.4 Deployment Architecture

```
DEPLOYMENT MODES

Mode 1: Batch Processing (CLI)
├── Command: python main.py video.mp4
├── Operation: Single video file processing
├── Output: Annotated video + metrics
└── Use Case: Post-match analysis

Mode 2: Web Dashboard (Streamlit)
├── Command: streamlit run streamlit_app.py
├── Operation: Web-based UI with real-time display
├── Output: Interactive visualization + metrics
└── Use Case: Live monitoring + historical review

Mode 3: Real-Time Streaming (Server + WebSocket)
├── Components:
│   ├── FastAPI Server (Port 8000)
│   ├── WebSocket Clients
│   └── Event Queue System
├── Operation: Continuous event publishing
├── Output: Real-time notifications + event log
└── Use Case: Live match monitoring + alerts
```

### 1.5 Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Object Detection** | YOLOv8 (Nano) | Real-time performance + accuracy balance |
| **Tracking** | SORT | Lightweight, efficient multi-object tracking |
| **Pose Estimation** | MediaPipe Pose | Fast inference, mobile-optimized, 33 joints |
| **Temporal Analysis** | LSTM (PyTorch) | Captures motion patterns over time |
| **Web Framework** | FastAPI | High-performance, async support |
| **UI Framework** | Streamlit | Rapid prototyping, interactive visualizations |
| **Logging** | Loguru | Structured, colored, timestamped logging |
| **Data Processing** | NumPy/OpenCV | Efficient numerical and image operations |
| **Computation** | CUDA (Optional) | GPU acceleration when available |

### 1.6 System Performance Characteristics

#### 1.6.1 Processing Efficiency

- **Frame Processing Rate:** 15-30 FPS (depending on hardware and model complexity)
- **Stage Latency:**
  - Stage 0 (Court): 2-3ms/frame
  - Stage 1 (Detection): 8-12ms/frame
  - Stage 2 (Raider ID): 5-8ms/frame
  - Stage 3 (Pose): 15-20ms/frame
  - Stage 4-7 (Analysis): 10-15ms/frame
- **Total Pipeline Latency:** ~60-80ms/frame

#### 1.6.2 Memory Requirements

- **Minimum:** 2GB RAM (CPU only)
- **Recommended:** 8GB RAM + GPU
- **Peak Usage:** ~4-6GB (GPU: 6-8GB VRAM)

#### 1.6.3 Scalability Characteristics

- **Concurrent Processes:** 4-8 independent video streams
- **Concurrent WebSocket Clients:** 50+ simultaneous connections
- **Database Capacity:** Supports 10,000+ player injury history records
- **Storage Requirements:** ~100MB-500MB per hour of video (HD resolution)

---

## Module Description

### 2.0 Module Architecture Overview

The system is organized into the following module categories:

1. **Core Pipeline Module** - Main orchestration and execution
2. **Detection Modules** - Entity detection and identification
3. **Analysis Modules** - Biomechanical and motion analysis
4. **Fusion Modules** - Risk aggregation and scoring
5. **Utility Modules** - Supporting functions and helpers
6. **Server Modules** - Real-time communication infrastructure
7. **Configuration Module** - Centralized parameter management

---

### 2.1 Core Pipeline Module

#### 2.1.1 KabaddiInjuryPipeline (main.py)

**Purpose:** Orchestrates the entire 7-stage processing pipeline, managing execution flow, resource allocation, and output generation.

**Key Classes:**

```python
class KabaddiInjuryPipeline:
    def __init__(self, video_path: str, output_dir: Optional[str] = None)
    def run(self) -> Dict[str, Any]
    def process_frame(frame: np.ndarray, frame_num: int) -> Dict
    def save_outputs() -> None
```

**Responsibilities:**
- Video file loading and validation
- Frame-by-frame pipeline orchestration
- Stage sequencing and error handling
- Model initialization and lifecycle management
- Output video generation and metric collection
- Server communication and event broadcasting
- Session tracking and logging

**Key Features:**

1. **Automatic Server Initialization**
   - Detects if FastAPI server is running
   - Starts server in background thread if needed
   - Validates server health before processing

2. **Real-Time Event Submission**
   - Raider identification events
   - Injury risk updates (throttled every 5 frames)
   - Fall detection events
   - Collision detection events

3. **Comprehensive Logging**
   - Timestamped log files
   - Debug-level file logging
   - Info-level console output
   - Log index for session tracking

4. **Progress Tracking**
   - Frame-by-frame progress bar
   - Stage completion tracking
   - Metrics aggregation
   - Performance metrics recording

**Input Parameters:**
- `video_path`: Path to input video file (MP4, AVI, MOV)
- `output_dir`: Optional output directory (default: ./outputs/session_*)
- `max_frames`: Optional frame limit for testing/debugging

**Output Data Structure:**

```python
{
    "session_id": str,
    "video_path": str,
    "output_dir": str,
    "frames_processed": int,
    "duration_seconds": float,
    "fps": float,
    "resolution": Tuple[int, int],
    "raider_id": Optional[int],
    "events": {
        "falls": List[Dict],
        "collisions": List[Dict],
        "risk_updates": List[Dict]
    },
    "metrics": {
        "avg_risk_score": float,
        "max_risk_score": float,
        "min_risk_score": float,
        "fall_count": int,
        "collision_count": int
    }
}
```

**Processing Flow:**

```
1. Initialize Pipeline
   ├── Validate video file
   ├── Create output directories
   ├── Initialize all models
   └── Start server (if available)

2. Main Processing Loop (frame-by-frame)
   ├── Read frame from video
   ├── Execute Stage 0: Court Detection
   ├── Execute Stage 1: Player Detection & Tracking
   ├── Execute Stage 2: Raider Identification
   ├── Execute Stage 3: Pose Estimation (raider only)
   ├── Execute Stage 4: Fall Detection
   ├── Execute Stage 5: Motion Analysis
   ├── Execute Stage 6: Impact Detection
   ├── Execute Stage 7: Risk Fusion
   ├── Annotate frame with results
   ├── Write to output video
   ├── Submit events to server (if applicable)
   └── Update metrics and logs

3. Finalization
   ├── Close video writers
   ├── Save metrics and summaries
   ├── Generate event timeline
   └── Create session report
```

---

### 2.2 Detection Modules

#### 2.2.1 CourtLineDetector (models/court_line_detector.py)

**Purpose:** Identifies and localizes court boundaries to provide spatial context for player analysis.

**Algorithm:** HSV Color Space + Morphological Operations + Hough Line Transform

**Key Methods:**

```python
class CourtLineDetector:
    def __init__(self)
    def detect_court_lines(frame: np.ndarray) -> Dict
    def _hsv_threshold(frame: np.ndarray) -> np.ndarray
    def _morphological_ops(binary_img: np.ndarray) -> np.ndarray
    def _detect_lines_hough(edges: np.ndarray) -> List[Tuple]
    def _categorize_lines(lines: List) -> Dict
    def get_court_coordinates() -> Dict
```

**Processing Steps:**

1. **Color Space Conversion:** RGB → HSV
2. **White Line Thresholding:**
   - Lower bound: (0, 0, 200)
   - Upper bound: (180, 40, 255)
3. **Morphological Operations:**
   - Closing to fill gaps
   - Dilation to enhance visibility
   - Kernel size: 5×5, 2 iterations
4. **Edge Detection:** Canny edge detector
   - Low threshold: 50
   - High threshold: 150
5. **Line Detection:** Hough Line Transform
   - Hough threshold: 50 votes
   - Minimum line length: 100 pixels
   - Maximum gap: 10 pixels
6. **Line Categorization:** Horizontal/Vertical classification
   - Angle tolerance: ±15°
   - Minimum separation: 50 pixels

**Output:**

```python
{
    "court_detected": bool,
    "lines": {
        "horizontal": List[Tuple[int, int, int, int]],
        "vertical": List[Tuple[int, int, int, int]],
        "all": List[Tuple[int, int, int, int]]
    },
    "court_bounds": {
        "top": int,
        "bottom": int,
        "left": int,
        "right": int
    },
    "confidence": float
}
```

**Performance:** 2-3ms per frame

---

#### 2.2.2 PlayerDetector (models/player_detector.py)

**Purpose:** Detects all human players in the frame and maintains consistent track IDs across frames.

**Architecture:** YOLOv8 (Nano model) + SORT Tracker

**Key Methods:**

```python
class PlayerDetector:
    def __init__(self)
    def detect_players(frame: np.ndarray) -> np.ndarray
    def track_players(detections: np.ndarray) -> List[Dict]
    def update_tracks(detections: np.ndarray) -> List[Dict]
    def get_tracked_players() -> List[Dict]
```

**Processing Pipeline:**

1. **YOLOv8 Inference**
   - Model: yolov8n.pt (Nano - optimized for speed)
   - Confidence threshold: 0.5
   - IoU threshold: 0.45
   - Device: Auto-detect (CUDA if available)

2. **Detection Format Conversion**
   - From: (x_min, y_min, x_max, y_max) format
   - To: (detection_array) for SORT tracker

3. **SORT Tracking**
   - Maximum age: 30 frames (tracks persist ~1 second)
   - Minimum hits: 3 detections before confirmation
   - IoU threshold: 0.3

**Output Per Player:**

```python
{
    "track_id": int,
    "bbox": [x1, y1, x2, y2],
    "center": (center_x, center_y),
    "confidence": float,
    "frame_num": int
}
```

**Performance:**
- Detections: 8-12ms per frame
- Tracking: 2-3ms per frame
- Supported device: CPU or CUDA

---

#### 2.2.3 RaiderIdentifier (models/raider_identifier.py)

**Purpose:** Identifies which player is the raider using multi-cue analysis without relying solely on visual features.

**Algorithm:** Multi-Cue Fusion (Court Logic + Motion Signature + Temporal Consistency)

**Key Methods:**

```python
class RaiderIdentifier:
    def __init__(self, frame_width: int, frame_height: int)
    def update_player_positions(players: List[Dict]) -> None
    def identify_raider(players: List[Dict], frame_num: int) -> Tuple[int, float]
    def _evaluate_raider_metrics() -> Dict[int, float]
    def _calculate_motion_signature(player_id: int) -> Dict
    def lock_raider(raider_id: int) -> None
```

**Identification Strategy:**

**Phase 1: Initial Detection (Frames 0-90)**
- Monitor player positions relative to court midline
- Identify crossing toward opponent's side (25% line)
- Track motion signature (velocity, acceleration patterns)
- Duration: ~3 seconds (90 frames at 30 FPS)

**Phase 2: Confirmation & Locking (Frame 90+)**
- Once raider identified, permanently lock the track ID
- Calculate confidence based on:
  - Crossing persistence (80% weight)
  - Motion signature consistency (15% weight)
  - Temporal stability (5% weight)
- Lock flag prevents raider switching

**Key Metrics:**

```
Motion Signature:
├── Continuous Presence: Player consistently in opponent territory
├── Velocity Pattern: Higher average velocity vs. defenders
├── Motion Range: Greater distance traveled per time window
└── Temporal Consistency: Similar patterns across evaluation window
```

**Output:**

```python
{
    "raider_id": int or None,
    "confidence": float (0-100),
    "locked": bool,
    "stage": str ("initializing" | "evaluating" | "locked"),
    "metrics": {
        "crossing_score": float,
        "motion_score": float,
        "temporal_score": float
    }
}
```

**Performance:** 5-8ms per frame

---

### 2.3 Analysis Modules

#### 2.3.1 PoseEstimator (models/pose_estimator.py)

**Purpose:** Extracts 33-point skeleton (body landmarks) from the raider's bounding box.

**Technology:** MediaPipe Pose (optimized for real-time performance)

**Key Methods:**

```python
class PoseEstimator:
    def __init__(self)
    def estimate_pose(frame: np.ndarray, bbox: List) -> Dict
    def extract_keypoints(detection_result: Any) -> Dict[str, Dict]
    def calculate_joint_angles(joints: Dict) -> Dict[str, float]
    def get_confidence_scores() -> Dict[str, float]
```

**33-Point Skeleton Structure:**

```
Head Region:
├── nose (0), left_eye (1), right_eye (2)
├── left_ear (3), right_ear (4)

Upper Body:
├── left_shoulder (11), right_shoulder (12)
├── left_elbow (13), right_elbow (14)
├── left_wrist (15), right_wrist (16)
├── left_pinky (17), right_pinky (18)
├── left_index (19), right_index (20)
├── left_thumb (21), right_thumb (22)

Torso:
├── mouth_left (9), mouth_right (10)

Lower Body:
├── left_hip (23), right_hip (24)
├── left_knee (25), right_knee (26)
├── left_ankle (27), right_ankle (28)
├── left_heel (29), right_heel (30)
├── left_foot_index (31), right_foot_index (32)
```

**Output Per Frame:**

```python
{
    "joints": {
        "joint_name": {
            "position": [x, y],
            "z": float,  # depth coordinate
            "confidence": float (0-1)
        },
        ...  # for all 33 joints
    },
    "visibility_score": float,
    "frame_quality": str ("good" | "poor" | "insufficient")
}
```

**Processing:**
- Input: Cropped frame (raider's bounding box)
- Inference library: MediaPipe Pose (TFLite backend)
- Confidence threshold: 0.5 (filters low-confidence landmarks)
- Performance: 15-20ms per frame

**Limitations & Handling:**
- Fails gracefully if raider occluded or partially visible
- Returns None for poor quality frames
- Temporal smoothing applied to reduce jitter

---

#### 2.3.2 FallDetector (models/fall_detector.py)

**Purpose:** Detects fall events and dangerous biomechanical positions using rule-based analysis.

**Algorithm:** Multi-Indicator Biomechanical Rule System

**Key Methods:**

```python
class FallDetector:
    def __init__(self, frame_height: int)
    def detect_fall(joints: Dict, frame_num: int) -> Tuple[bool, Dict]
    def _get_hip_height(joints: Dict) -> float
    def _calculate_torso_angle(joints: Dict) -> float
    def _estimate_head_velocity(joints: Dict) -> float
    def _check_ground_contact(joints: Dict) -> bool
```

**Fall Detection Indicators (5 Metrics):**

1. **Hip Drop Indicator**
   - Threshold: Hip Y position drops > 15% of frame height in 3 frames
   - Weight: 25%
   - Indicates: Loss of vertical stability

2. **Torso Tilt Indicator**
   - Threshold: Torso angle from vertical > 35°
   - Weight: 25%
   - Indicates: Loss of body alignment

3. **Head Velocity Indicator**
   - Threshold: Head velocity > 2 pixels/frame sustained
   - Weight: 20%
   - Indicates: Uncontrolled motion

4. **Ground Contact Indicator**
   - Threshold: Knee/ankle Y coordinate near frame bottom (>85% frame height)
   - Weight: 20%
   - Indicates: Body contact with ground/floor

5. **Sudden Stop Indicator**
   - Threshold: Velocity drops to near zero after high motion
   - Weight: 10%
   - Indicates: Impact/collision

**Severity Calculation:**

```
Fall Severity (0-100) = 
    (hip_drop_score × 0.25) +
    (torso_tilt_score × 0.25) +
    (head_velocity_score × 0.20) +
    (ground_contact_score × 0.20) +
    (sudden_stop_score × 0.10)
```

**Output:**

```python
{
    "is_falling": bool,
    "fall_severity": float (0-100),
    "indicators": {
        "hip_drop": bool,
        "torso_tilt": bool,
        "head_velocity": bool,
        "ground_contact": bool,
        "sudden_stop": bool
    },
    "metrics": {
        "hip_height": float,
        "torso_angle": float,
        "head_velocity": float
    },
    "frame_num": int
}
```

**Performance:** 5-8ms per frame

**Event Recording:** Fall events stored in timeline with:
- Start frame number
- Peak severity score
- Duration (frames)
- Location in video
- Contributing indicators

---

#### 2.3.3 MotionAnalyzer (models/motion_analyzer.py)

**Purpose:** Analyzes temporal motion patterns and identifies abnormal biomechanical signatures.

**Architecture:** Hybrid (Rule-Based + LSTM Deep Learning)

**Key Methods:**

```python
class MotionAnalyzer:
    def __init__(self)
    def update_joint_data(joints: Dict) -> None
    def analyze_motion(frame_num: int) -> Dict
    def calculate_motion_metrics() -> Dict[str, float]
    def detect_motion_abnormality() -> float
```

**Motion Metrics Computed:**

1. **Velocity Analysis**
   - Per-joint velocity vectors
   - Average velocity across body
   - Velocity asymmetry (left vs. right side)

2. **Acceleration Analysis**
   - Joint-level acceleration
   - Sudden acceleration detection
   - Jerk (rate of change of acceleration)

3. **Range of Motion**
   - Joint displacement magnitude
   - Movement fluidity score
   - Restricted motion detection

4. **Temporal Patterns**
   - 90-frame evaluation window (~3 seconds)
   - Sliding window overlap: 50%
   - Pattern consistency scoring

**LSTM Component (Temporal Motion Analyzer):**

```python
class MotionLSTM(nn.Module):
    Architecture:
    ├── Input: 18-dimensional joint feature vector
    ├── LSTM Cell 1: 128 hidden units
    ├── LSTM Cell 2: 128 hidden units (with dropout)
    ├── Fully Connected 1: 128 → 64
    ├── ReLU Activation
    ├── Dropout (0.3)
    └── Output: Single abnormality score (0-1)
```

**Output:**

```python
{
    "motion_abnormality_score": float (0-100),
    "velocity_metrics": {
        "avg_velocity": float,
        "max_velocity": float,
        "velocity_asymmetry": float
    },
    "acceleration_metrics": {
        "avg_acceleration": float,
        "max_jerk": float
    },
    "range_metrics": {
        "total_displacement": float,
        "movement_fluidity": float
    },
    "lstm_score": float (0-100) if available,
    "frame_num": int
}
```

**Performance:** 10-15ms per frame (rule-based), 8-12ms additional for LSTM

---

#### 2.3.4 ImpactDetector (models/impact_detector.py)

**Purpose:** Detects defender-raider interactions and models collision intensity without requiring full pose estimation of defenders.

**Algorithm:** Proximity-Based Physics Model + Velocity Vector Analysis

**Key Methods:**

```python
class ImpactDetector:
    def __init__(self)
    def update_defender_positions(players: List[Dict], raider_id: int) -> None
    def detect_collisions(raider_bbox: List, frame_num: int) -> List[Dict]
    def calculate_defender_velocity(defender_id: int) -> np.ndarray
    def estimate_collision_energy() -> float
```

**Collision Detection Strategy:**

1. **Proximity Analysis**
   - Collision radius: 80 pixels (configurable)
   - Approach radius: 150 pixels
   - Detects when defender enters collision zone

2. **Velocity-Based Impact Scoring**
   - Calculate defender velocity from bounding box tracking
   - Impact energy: proportional to velocity magnitude
   - Formula: `Impact Score = defender_velocity × proximity_factor`

3. **Multi-Defender Tracking**
   - Tracks all non-raider players simultaneously
   - Computes individual impact scores
   - Aggregates total collision impact

**Collision Event Structure:**

```python
{
    "collision_frame": int,
    "raider_id": int,
    "defender_ids": List[int],
    "defender_count": int,
    "collision_location": [x, y],
    "collision_energy": float (0-100),
    "defender_velocities": Dict[int, float],
    "duration_frames": int,
    "severity": str ("minor" | "moderate" | "severe")
}
```

**Impact Energy Calculation:**

```
Impact Score (0-100) = 
    min(100, 
        sum(defender_velocity[i] × proximity_factor[i]) 
        for all defenders in collision zone
    )
```

**Performance:** 8-10ms per frame

---

### 2.4 Fusion Module

#### 2.4.1 RiskFusionEngine (models/risk_fusion.py)

**Purpose:** Aggregates multiple risk factors into a single, interpretable injury risk score.

**Algorithm:** Weighted Multi-Factor Fusion with Temporal Smoothing

**Key Methods:**

```python
class RiskFusionEngine:
    def __init__(self)
    def compute_risk_score(
        fall_score: float,
        motion_score: float,
        impact_score: float,
        raider_id: int
    ) -> Tuple[float, Dict]
    def get_injury_history_modifier(player_id: int) -> float
    def smooth_risk_score(raw_score: float) -> float
    def classify_risk_level(score: float) -> str
```

**Risk Component Weights:**

| Component | Weight | Scale | Source |
|-----------|--------|-------|--------|
| Fall Score | 40% | 0-100 | FallDetector |
| Motion Abnormality | 30% | 0-100 | MotionAnalyzer |
| Collision Impact | 20% | 0-100 | ImpactDetector |
| Injury History | 10% | 0-100 | CSV Database |

**Risk Score Calculation:**

```
Raw Risk Score = 
    (fall_score × 0.40) +
    (motion_score × 0.30) +
    (impact_score × 0.20) +
    (injury_history × 0.10)

Final Risk Score = Temporal_Smoothing(Raw Risk Score)
```

**Temporal Smoothing (Kalman-like Filter):**

```
Smoothed Score = (Raw Score × 0.4) + (Previous Score × 0.6)
Window Size: 10 frames
```

**Risk Level Classification:**

| Score Range | Level | Color | Interpretation |
|-------------|-------|-------|-----------------|
| 0-25 | LOW | Green | Minimal injury risk |
| 25-50 | MODERATE | Yellow | Elevated risk, monitoring recommended |
| 50-75 | HIGH | Orange | Significant risk, intervention advised |
| 75-100 | CRITICAL | Red | Severe risk, immediate intervention required |

**Injury History Database:**

```python
Schema: injury_history.csv
├── player_id: Unique identifier
├── past_injuries: Count of previous injuries
├── recent_injury_days: Days since last injury
├── severity_history: Average severity of past injuries
├── risk_multiplier: Cumulative risk elevation factor
└── last_updated: Timestamp
```

**Output:**

```python
{
    "injury_risk_score": float (0-100),
    "risk_level": str ("LOW" | "MODERATE" | "HIGH" | "CRITICAL"),
    "components": {
        "fall_contribution": float,
        "motion_contribution": float,
        "impact_contribution": float,
        "history_contribution": float
    },
    "temporal_smoothed": bool,
    "confidence": float (0-100),
    "timestamp": float,
    "frame_num": int
}
```

**Performance:** 8-10ms per computation

---

### 2.5 Utility Modules

#### 2.5.1 VideoUtils (utils/video_utils.py)

**Purpose:** High-level abstraction for video I/O operations with frame caching and format detection.

**Key Classes:**

```python
class VideoReader:
    def __init__(self, video_path: str)
    def read_frame(frame_num: int) -> np.ndarray
    def get_fps() -> float
    def get_resolution() -> Tuple[int, int]
    def get_total_frames() -> int
    def close() -> None

class VideoWriter:
    def __init__(self, output_path: str, fps: float, resolution: Tuple[int, int])
    def write_frame(frame: np.ndarray) -> None
    def close() -> None
```

**Features:**
- Automatic codec detection
- Resolution validation
- Error handling for corrupted frames
- Memory-efficient frame buffering
- Support for multiple formats (MP4, AVI, MOV, etc.)

---

#### 2.5.2 Visualization (utils/visualization.py)

**Purpose:** Frame annotation and visual rendering of detection results, poses, and risk scores.

**Key Methods:**

```python
class VideoAnnotator:
    def annotate_bounding_boxes(frame: np.ndarray, detections: Dict) -> np.ndarray
    def annotate_skeleton(frame: np.ndarray, joints: Dict) -> np.ndarray
    def annotate_fall_indicator(frame: np.ndarray, fall_info: Dict) -> np.ndarray
    def annotate_risk_score(frame: np.ndarray, risk_info: Dict) -> np.ndarray
    def annotate_collision_zone(frame: np.ndarray, raider_bbox: List) -> np.ndarray
```

**Visualization Elements:**
- Bounding boxes with player IDs and confidence
- Skeleton visualization (raider only)
- Fall event indicators (red box, severity label)
- Risk score gauge (0-100 colored bar)
- Collision warning zones
- Event timeline overlay

---

#### 2.5.3 PipelineStatus (utils/pipeline_status.py)

**Purpose:** Tracks and reports progress of pipeline execution with detailed stage metrics.

**Key Classes:**

```python
class PipelineStatus:
    def __init__(self, output_dir: Path)
    def update_stage(stage_num: int, status: str, details: str) -> None
    def get_stage_status(stage_num: int) -> Dict
    def get_overall_status() -> Dict
    def save_status_report() -> None

class MetricsLogger:
    def __init__(self, output_dir: Path)
    def log_metric(name: str, value: Any) -> None
    def finalize() -> None
```

**Tracked Metrics:**
- Frames processed per stage
- Processing time per stage
- Error counts and types
- Detection/tracking statistics
- Risk score statistics (mean, max, min)
- Event counts (falls, collisions)

---

#### 2.5.4 EventLogger & RaiderStatusTracker (utils/)

**Purpose:** Centralized event logging and raider state management across pipeline execution.

**Key Methods:**

```python
class EventLogger:
    def log_fall_event(event: Dict) -> None
    def log_collision_event(event: Dict) -> None
    def log_risk_update(risk_info: Dict) -> None
    def export_timeline(output_path: str) -> None

class RaiderStatusTracker:
    def update_status(frame_num: int, raider_id: int, status: Dict) -> None
    def get_current_status() -> Dict
    def get_status_history() -> List[Dict]
```

---

### 2.6 Server Module

#### 2.6.1 FastAPI Server (server.py)

**Purpose:** Real-time event streaming and WebSocket client management for live monitoring.

**Architecture:** Asynchronous FastAPI with WebSocket support

**Key Endpoints:**

```
REST API Endpoints:

POST /event/raider-identified
├── Purpose: Notify of raider identification
├── Payload: {"raider_id": int, "frame": int, "confidence": float}
├── Response: {"status": "received"}
└── Broadcast: WebSocket event to all connected clients

POST /event/injury-risk
├── Purpose: Submit injury risk score update
├── Payload: {"risk_score": float, "frame": int, "components": {...}}
├── Response: {"status": "received"}
└── Broadcast: WebSocket event to all connected clients

POST /event/collision
├── Purpose: Notify of collision event
├── Payload: {"raider_id": int, "defenders": [...], "energy": float, "frame": int}
├── Response: {"status": "received"}
└── Broadcast: WebSocket event to all connected clients

POST /event/fall
├── Purpose: Notify of fall detection
├── Payload: {"severity": float, "frame": int, "indicators": {...}}
├── Response: {"status": "received"}
└── Broadcast: WebSocket event to all connected clients

GET /health
├── Purpose: Health check and status monitoring
├── Response: {"status": "healthy", "timestamp": ..., "connected_clients": int}
└── Use Case: Server availability verification

WS /ws
├── Purpose: WebSocket connection for real-time event stream
├── Features: Persistent connection, bidirectional communication
├── Message Format: JSON event objects
└── Client Auto-Response: PING → PONG keep-alive
```

**WebSocket Event Message Format:**

```python
{
    "type": str,  # "RAIDER_IDENTIFIED" | "INJURY_RISK" | "COLLISION" | "FALL"
    "data": Dict,  # Event-specific payload
    "server_timestamp": str,  # ISO 8601 timestamp
    "client_id": Optional[str]  # For client identification
}
```

**Key Classes:**

```python
class FastAPIServer:
    async def health_check() -> Dict
    async def raider_identified(data: Dict) -> Dict
    async def injury_risk_update(data: Dict) -> Dict
    async def collision_event(data: Dict) -> Dict
    async def fall_event(data: Dict) -> Dict
    async def websocket_endpoint(websocket: WebSocket) -> None
    async def broadcast_event(event: Dict) -> None
```

**Features:**

1. **Thread-Safe Client Management**
   - `clients_lock` for concurrent access
   - Safe add/remove operations
   - Atomic broadcast

2. **Event Queue**
   - `event_queue` (max 100 events)
   - Prevents memory overflow
   - FIFO processing

3. **Middleware**
   - CORS enabled for all origins
   - Request validation
   - Error handling and logging

4. **PING/PONG Keep-Alive**
   - Automatic heartbeat every 30 seconds
   - Detects stale connections
   - Automatic reconnection on client side

---

#### 2.6.2 Streamlit Dashboard (streamlit_app.py)

**Purpose:** Web-based interactive dashboard for video upload, processing, and results visualization.

**Key Features:**

1. **Video Management**
   - Drag-and-drop upload
   - File validation
   - Format support: MP4, AVI, MOV, WebM

2. **Real-Time Processing Display**
   - Progress bar with frame counter
   - Stage-by-stage status updates
   - Live event notifications

3. **Results Visualization**
   - Annotated video playback
   - Interactive risk score chart
   - Fall/collision event timeline
   - Metrics dashboard

4. **Historical Analysis**
   - Session history browser
   - Comparative metrics
   - Player tracking across sessions

5. **Server Integration**
   - Auto-start server thread
   - Health monitoring
   - WebSocket client library

---

### 2.7 Configuration Module

#### 2.7.1 Config (config/config.py)

**Purpose:** Centralized parameter management for all models, thresholds, and system constants.

**Structure:**

```python
# Project paths
BASE_DIR, DATA_DIR, OUTPUT_DIR, MODELS_DIR
STAGE_DIRS, INJURY_HISTORY_PATH

# Stage 0: Court Detection
COURT_LINE_CONFIG = {
    "hsv_lower": (0, 0, 200),
    "hsv_upper": (180, 40, 255),
    "canny_low": 50,
    "canny_high": 150,
    # ... additional parameters
}

# Stage 1: Detection & Tracking
YOLO_CONFIG = {
    "model": "yolov8n.pt",
    "confidence": 0.5,
    "device": "cuda" | "cpu"  # auto-detect
}

TRACKING_CONFIG = {
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3
}

# Stage 2: Raider Identification
RAIDER_CONFIG = {
    "midline_position": 0.5,
    "crossing_threshold": 0.3,
    "evaluation_window": 90,  # frames
    # ... additional parameters
}

# Stage 3-6: Model Parameters
POSE_CONFIG, FALL_CONFIG, MOTION_CONFIG, IMPACT_CONFIG

# Stage 7: Risk Fusion
RISK_FUSION_CONFIG = {
    "weights": {
        "fall": 0.40,
        "motion": 0.30,
        "impact": 0.20,
        "history": 0.10
    },
    "risk_levels": {
        "low": (0, 25),
        "moderate": (25, 50),
        "high": (50, 75),
        "critical": (75, 100)
    },
    "temporal_smoothing": True,
    "smoothing_window": 10
}

# Video processing
VIDEO_CONFIG = {
    "fps": 30,
    "max_resolution": (1280, 720),
    "codec": "mp4v"
}

# Logging
LOG_CONFIG = {
    "level": "DEBUG",
    "format": "%(timestamp)s | %(level) | %(message)s"
}

# Performance
PERFORMANCE_CONFIG = {
    "cuda_enabled": torch.cuda.is_available(),
    "max_batch_size": 32,
    "frame_buffer_size": 10
}
```

**Benefits:**
- Single source of truth for all parameters
- Easy parameter tuning without code changes
- Environment-aware defaults (CUDA detection)
- Clear documentation of ranges and constraints

---

### 2.8 Complete Module Dependency Graph

```
┌─────────────────────────────────────────┐
│         main.py (Orchestrator)          │
│    (KabaddiInjuryPipeline)              │
└──────────┬──────────────────────────────┘
           │
    ┌──────┴───────────────────────────────────────────────────┐
    │                                                            │
    ▼                                                            ▼
┌──────────────────────┐                          ┌────────────────────┐
│  Detection Modules   │                          │ Analysis Modules   │
├──────────────────────┤                          ├────────────────────┤
│ - CourtLineDetector  │     ┌──────────────┐    │ - PoseEstimator    │
│ - PlayerDetector     │◄────│config.config │    │ - FallDetector     │
│ - RaiderIdentifier   │     └──────────────┘    │ - MotionAnalyzer   │
└──────────────────────┘                          │ - ImpactDetector   │
                                                  └────────────────────┘
                                                          │
                                                          ▼
                                                  ┌────────────────────┐
                                                  │  Fusion Module     │
                                                  ├────────────────────┤
                                                  │ - RiskFusionEngine │
                                                  └──────────┬─────────┘
                                                             │
                         ┌───────────────────────────────────┤
                         │                                    │
                         ▼                                    ▼
                  ┌─────────────────┐              ┌────────────────────┐
                  │ Utility Modules │              │ Server Infrastructure
                  ├─────────────────┤              ├────────────────────┤
                  │ - VideoUtils    │              │ - FastAPI Server   │
                  │ - Visualization │              │ - WebSocket Support│
                  │ - PipelineStatus│              │ - Client Manager   │
                  │ - EventLogger   │              │                    │
                  │ - RaiderTracker │              │ - Streamlit App    │
                  └─────────────────┘              └────────────────────┘
```

---

### 2.9 Module Interaction Sequences

#### Processing Sequence (Per Frame)

```
Frame N Processing:

1. VideoReader.read_frame(N)
   └─▶ Returns raw frame (H, W, 3)

2. CourtLineDetector.detect_court_lines(frame)
   └─▶ Court metadata (for context, not used in other stages)

3. PlayerDetector.detect_players(frame)
   └─▶ Person detections [[x1, y1, x2, y2, conf], ...]

4. PlayerDetector.track_players(detections)
   ├─▶ SORTTracker.update(detections)
   └─▶ Tracked players with IDs [{id, bbox, center}, ...]

5. RaiderIdentifier.update_player_positions(tracked_players)
   └─▶ Updates position history

6. RaiderIdentifier.identify_raider(tracked_players, frame_num)
   ├─▶ Evaluates crossing and motion metrics
   ├─▶ If raider_locked: returns raider_id
   └─▶ Else: continues evaluation

7. If raider identified:
   
   7.1. PoseEstimator.estimate_pose(frame, raider_bbox)
        └─▶ 33-joint skeleton {joint_name: {position, confidence}}
   
   7.2. FallDetector.detect_fall(joints, frame_num)
        └─▶ Fall indicators + severity score
   
   7.3. MotionAnalyzer.update_joint_data(joints)
        └─▶ Updates temporal buffers
   
   7.4. MotionAnalyzer.analyze_motion()
        └─▶ Motion abnormality score + metrics

8. ImpactDetector.update_defender_positions(tracked_players, raider_id)
   └─▶ Updates defender tracking

9. ImpactDetector.detect_collisions(raider_bbox, frame_num)
   └─▶ Collision events + impact scores

10. RiskFusionEngine.compute_risk_score(fall, motion, impact, raider_id)
    ├─▶ Loads injury history for raider
    ├─▶ Computes raw weighted score
    ├─▶ Applies temporal smoothing
    └─▶ Classifies risk level

11. VideoAnnotator.annotate_*()
    ├─▶ Bounding boxes
    ├─▶ Skeleton (if raider visible)
    ├─▶ Fall indicators
    ├─▶ Risk score
    └─▶ Returns annotated frame

12. VideoWriter.write_frame(annotated_frame)
    └─▶ Writes to output video

13. EventLogger.log_*()
    ├─▶ If fall: log_fall_event()
    ├─▶ If collision: log_collision_event()
    └─▶ Always: log_risk_update()

14. Server communication (if enabled):
    ├─▶ Raider event (once per identification)
    ├─▶ Collision events (as they occur)
    ├─▶ Fall events (as they occur)
    └─▶ Risk updates (every 5 frames, throttled)
```

---

## Conclusion

The Kabaddi Injury Prediction System represents a comprehensive, production-ready solution for real-time injury risk assessment in sports environments. Its modular architecture, explainable AI approach, and hybrid processing methodology provide a robust foundation for both immediate deployment and future enhancements.

The system successfully demonstrates:
- **Technical Excellence:** State-of-the-art computer vision and deep learning integration
- **Practical Deployability:** Multiple deployment modes (CLI, Web, Streaming)
- **Interpretability:** Clear, traceable risk components understandable to domain experts
- **Scalability:** Efficient processing pipeline capable of handling high-resolution video streams
- **Extensibility:** Modular design allowing easy incorporation of new analysis methods

---

**Document Prepared By:** Development Team  
**Date:** March 2026  
**Document Version:** 1.0  
**Status:** Final Review Ready

# config.py - Documentation

## 1. Overview
`config/config.py` acts as the central source of truth for all system configurations. It isolates hardcoded values, thresholds, and paths from the logic code, making the system easy to tune and maintain. It defines parameters for every stage of the pipeline, from neural network model paths to risk calculation weights.

## 2. Technical Overview
- **Type**: Configuration Module
- **Dependencies**: `pathlib`, `torch` (for device detection).
- **Structure**: A set of global uppercase dictionaries and constants.

## 3. Detailed Technical
- **Path Management**: Uses `pathlib` for OS-agnostic paths. Automatically creates necessary directories (`data`, `outputs` and sub-stage folders) when the module is imported (or when directories are referenced).
- **Hardware Detection**: Automatically sets `DEVICE` to "cuda" if `torch.cuda.is_available()`, else "cpu".

## 4. Configuration Groups

### Project Paths
- `BASE_DIR`, `DATA_DIR`, `OUTPUT_DIR`: Root paths.
- `STAGE_DIRS`: Dictionary mapping stage IDs (1-7) to their specific output folders.
- `INJURY_HISTORY_PATH`: Path to the CSV database of player injury history.

### Stage Configurations
- **`YOLO_CONFIG` (Stage 1)**: Model path (`yolov8n.pt`), confidence thresholds, class IDs.
- **`RAIDER_CONFIG` (Stage 2)**: Rules for identifying raiders (speed, court position, consistency).
- **`POSE_CONFIG` (Stage 3)**: MediaPipe parameters (model complexity, smoothing).
- **`FALL_CONFIG` (Stage 4)**: Physics-based thresholds for fall detection (vertical drop speed, orientation change angle).
- **`MOTION_CONFIG` (Stage 5)**: Thresholds for velocity/acceleration/jerk and abnormality weights.
- **`IMPACT_CONFIG` (Stage 6)**: Proximity and velocity thresholds for collision detection.
- **`RISK_FUSION_CONFIG` (Stage 7)**:
    - **Weights**: Importance of each factor (Fall: 0.35, Impact: 0.30, Motion: 0.25, History: 0.10).
    - **Levels**: Risk score ranges (Low: 0-30, Medium: 31-70, High: 71-100).
    - **Thresholds**: Alert levels.

### Other
- **`VIS_CONFIG`**: Colors and visual styles for video annotation (BGR tuples).
- **`VIDEO_CONFIG`**: Output video settings (FPS, Codec).
- **`LOG_CONFIG`**: Logging format and levels.

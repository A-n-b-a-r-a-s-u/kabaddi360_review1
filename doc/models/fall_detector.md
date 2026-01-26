# fall_detector.py - Documentation

## 1. Overview
`models/fall_detector.py` implements **Stage 4** of the pipeline. It uses deterministic biomechanical rules to identify when the raider falls. Falls are a primary indicator of potential injury. By tracking key body parts (hips, head, torso) over time, it determines if a sudden, uncontrolled descent has occurred.

## 2. Technical Overview
- **Type**: Rule-based Biomechanical Analyzer
- **Dependencies**: `numpy`, `collections.deque`.
- **Input**: Raider joint positions (from Stage 3), Center of Mass.
- **Output**: `fall_info` (Dictionary with `is_falling`, `severity`, `indicators`).

## 3. Detailed Technical
The module checks 5 distinct indicators simultaneously. A fall is confirmed if a minimum number of these indicators (usually 2) are active.
1.  **Hip Drop**: Rapid vertical descent of the hip center.
2.  **Torso Tilt**: Sudden change in torso angle vs vertical (e.g., going from upright to horizontal).
3.  **Head Velocity**: Downward velocity of the nose landmark.
4.  **Ground Contact**: Hips staying near the bottom of the frame for a sustained duration.
5.  **Sudden Stop**: High deceleration of the center of mass (hitting the ground).

## 4. Workflow and Function Details

### Workflow
1.  **Data Ingestion**: Receives joints and frame number.
2.  **Update History**: Appends current positions to sliding window buffers (deque).
3.  **Check Rules**: Evaluates each of the 5 indicators against thresholds in `FALL_CONFIG`.
4.  **Scoring**: Adds up severity points for triggered indicators.
5.  **Confirmation**: Sets `is_falling = True` if condition count >= threshold.

### Functions

#### `FallDetector` Class

##### `detect_fall(self, joints, frame_num, com) -> (bool, dict)`
- **Purpose**: Main logic.
- **Details**: 
    - Computes metrics (speed, angle).
    - Checks thresholds defined in `config.py`.
    - Returns boolean flag and detailed breakdown.

##### `_get_hip_height(self, joints)`
- **Purpose**: Helper.
- **Details**: Averages Left and Right Hip Y-coordinates.

##### `_calculate_torso_angle(self, joints)`
- **Purpose**: Helper.
- **Details**: Computes angle of the vector [HipCenter -> Nose] against the vertical axis [0, -1].

##### `annotate_fall_detection(frame, fall_info, ...)`
- **Purpose**: Visualization.
- **Details**: Draws a "FALL DETECTED" alert on the frame if active. Lists triggered indicators (e.g., "Hip Drop: YES").

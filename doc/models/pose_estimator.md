# pose_estimator.py - Documentation

## 1. Overview
`models/pose_estimator.py` implements **Stage 3** of the pipeline. Once the raider is identified, this module extracts their skeletal structure. It focuses specifically on the raider to save computational resources. It estimates the positions of 33 key body landmarks (shoulders, hips, knees, ankles, etc.) which are essential for fall and injury analysis.

## 2. Technical Overview
- **Type**: Computer Vision / Machine Learning Module
- **Dependencies**: 
    - `mediapipe`: Google's framework for high-fidelity pose tracking.
    - `cv2`, `numpy`.
    - `utils.tracking_utils.LandmarkKalmanFilter`: For noise reduction.
- **Input**: Full frame + Raider Bounding Box.
- **Output**: 
    - `joints`: Dictionary of keypoints with smoothed (x,y) coordinates.
    - `angles`: Dictionary of biomechanical angles (knee flexion, torso lean).

## 3. Detailed Technical
- **ROI Cropping**: To improve accuracy and speed, it crops the frame to the raider's bounding box (with margin) before passing it to MediaPipe. This effectively "zooms in" on the relevant person.
- **Smoothing**: Raw MediaPipe output can be shaky. The module applies a **Kalman Filter** to each joint independently. This predicts the next state and corrects it based on measurement, resulting in smooth trajectories even if detection jitters.
- **Angle Calculation**: Computes geometric angles using dot product logic for key joints (e.g., Hip-Knee-Ankle for knee flexion).

## 4. Workflow and Function Details

### Workflow
1.  **Crop**: Extract image region defined by `bbox`.
2.  **Infer**: Run MediaPipe Pose on the crop.
3.  **Map**: Convert crop-relative coordinates back to full-frame global coordinates.
4.  **Filter**: Update Kalman filters for each joint.
5.  **Compute**: Calculate angles and velocities.

### Functions

#### `PoseEstimator` Class

##### `extract_pose(self, frame, bbox) -> dict`
- **Purpose**: Main extraction function.
- **Details**: 
    - Crops image.
    - Calls `self.pose.process()`.
    - Iterates over `KEY_JOINTS` config.
    - Applies `self.joint_filters[name].update()`.
    - Updates history buffers.

##### `calculate_joint_angles(self, joints)`
- **Purpose**: Biomechanical analysis.
- **Details**: Computes:
    - `left/right_knee_angle`: Critical for ACL injury risk.
    - `torso_angle`: Lean from vertical (critical for head-first falls).
    - Uses helper `_calculate_angle`.

##### `_initialize_filters(self)`
- **Purpose**: Setup.
- **Details**: Creates `LandmarkKalmanFilter` instances for specific joints defined in config (shoulders, hips, knees, ankles, etc.).

##### `annotate_pose(frame, joints, angles)`
- **Purpose**: Visualization helper.
- **Details**: Draws the skeleton stick-figure connecting joints. Displays calculated angles as text overlay.

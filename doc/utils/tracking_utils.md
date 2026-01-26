# tracking_utils.py - Documentation

## 1. Overview
`utils/tracking_utils.py` contains the core algorithms for object tracking and signal smoothing. It implements the **SORT (Simple Online and Realtime Tracking)** algorithm to keep track of players across frames, and **Kalman Filters** to smooth out jittery detection data.

## 2. Technical Overview
- **Type**: Algorithm Implementation Module
- **Dependencies**: 
    - `filterpy`: Kalman Filter implementation.
    - `scipy`: Hungarian algorithm (linear assignment) for matching.
    - `numpy`.
- **Algorithms**:
    - **Kalman Filter**: A recursive estimator that predicts the future state of a system (position/velocity) based on noisy measurements.
    - **SORT**: A tracking-by-detection framework that associates detections to tracks using IoU optimization.

## 3. Detailed Technical
- **KalmanBoxTracker**:
    - State Vector: `[u, v, s, r, u_dot, v_dot, s_dot]` (Center X, Center Y, Scale, Ratio, and their velocities).
    - It predicts where a bounding box *should* be in the next frame.
- **SORTTracker**:
    - matches YOLO detections to existing Kalman trackers.
    - Uses the Hungarian Algorithm to solve the assignment problem (minimize total distance/maximize total IoU).
    - Handles track creation (new player enters) and deletion (player leaves/occluded).
- **LandmarkKalmanFilter**:
    - Specialized filter for 2D points (Pose Landmarks).
    - Smoothes the `(x, y)` coordinates of joints to prevent "shaking" in the skeleton visualization.

## 4. Functions

### `SORTTracker` Class

##### `update(self, detections) -> tracks`
- **Purpose**: Main tracking loop.
- **Details**: 
    1. Predicts new locations of all existing tracks.
    2. Computes IoU matrix between predictions and new detections.
    3. Solves assignment.
    4. Updates matched tracks.
    5. Returns active track identities.

### `LandmarkKalmanFilter` Class

##### `update(self, measurement)`
- **Purpose**: Smoothing.
- **Details**: Takes a noisy point `(x, y)`, returns a smoothed point `(x_smooth, y_smooth)`.

##### `get_velocity(self)`
- **Purpose**: Kinematics.
- **Details**: Returns the velocity vector `(vx, vy)` estimated by the internal state.

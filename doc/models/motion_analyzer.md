# motion_analyzer.py - Documentation

## 1. Overview
`models/motion_analyzer.py` implements **Stage 5** of the pipeline. It evaluates the quality and safety of the raider's movement. It calculates derivatives of position (Velocity, Acceleration, Jerk) to quantify joint stress. It also computes an "abnormality score" which indicates how erratic or dangerous the movement is, potentially using an LSTM neural network alongside rule-based metrics.

## 2. Technical Overview
- **Type**: Motion Physics & ML Hybrid Module
- **Dependencies**: 
    - `numpy`
    - `models.temporal_lstm` (optional, for sequences).
- **Input**: Joint positions history.
- **Output**: `metrics` (Velocity/Accel per joint), `abnormality_score` (0-100).

## 3. Detailed Technical
- **Kinematics**: Uses finite difference method to calculate Velocity (1st derivative), Acceleration (2nd), and Jerk (3rd). High jerk is a strong correlate of injury risk (unsmooth motion).
- **Asymmetry**: Compares Left vs Right side metrics (e.g., Left Knee Accel vs Right Knee Accel). Significant imbalance (>30%) suggests injury or favoring a limb.
- **Smoothness**: Quantifies how "fluid" the movement is.
- **Hybrid Scoring**: Combines rule-based penalties (exceeding thresholds) with a Deep Learning (LSTM) model's prediction of "normal vs abnormal" motion sequences, if the model weighs file is present.

## 4. Workflow and Function Details

### Workflow
1.  **Update Buffers**: Store latest joint positions.
2.  **Compute Kinematics**: Calculate V, A, J for all joints.
3.  **Compute Metrics**: Aggregate max/avg/std values.
4.  **Compute Score**: 
    - Rule-based: Sum weighted penalties for threshold violations.
    - LSTM (if active): Inferences on the position sequence.
    - Weighted average of both.

### Functions

#### `MotionAnalyzer` Class

##### `calculate_motion_metrics(self, joints) -> dict`
- **Purpose**: Computes physics metrics.
- **Details**: Iterates all joints. Calculates Jerk. Calls `_calculate_smoothness` and `_calculate_asymmetry`.

##### `calculate_abnormality_score(self, metrics) -> float`
- **Purpose**: Generates the single 0-100 score.
- **Details**: Normalizes metrics against their config thresholds (e.g., `vel / vel_threshold`). Sums them up based on `abnormality_weights`.

##### `_calculate_smoothness(self, positions)`
- **Purpose**: Mathematical smoothness.
- **Details**: Uses the negative magnitude of average jerk (lower jerk = higher smoothness).

##### `_reconstruct_joints_from_buffer(self)`
- **Purpose**: Data formatting.
- **Details**: Prepares the deque data into the dictionary format expected by the LSTM model.

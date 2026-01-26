# temporal_lstm.py - Documentation

## 1. Overview
`models/temporal_lstm.py` provides an advanced deep learning capability to the Motion Analysis stage. While `motion_analyzer.py` uses physics rules (velocity, jerk), this module uses a **Long Short-Term Memory (LSTM)** neural network to recognize complex, non-linear abnormal motion patterns over time (e.g., a specific sequence of limb movements that precedes an injury) which might be missed by simple thresholding.

## 2. Technical Overview
- **Type**: Deep Learning Module (PyTorch)
- **Dependencies**: `torch`, `torch.nn`, `numpy`.
- **Input**: Sequence of 18 joint coordinates (9 joints × 2 xy) over 30 frames.
- **Output**: Abnormality Probability (0.0 - 1.0).

## 3. Detailed Technical
- **Architecture**:
    - **Input Layer**: 18 features (Nose, Shoulders, Hips, Knees, Ankles).
    - **LSTM Layers**: 2 stacked LSTM layers with 128 hidden units. This captures temporal dependencies.
    - **Dropout**: 30% dropout for regularization.
    - **Heads**: Fully connected layers ending in a Sigmoid activation to output a probability of "abnormality".
- **Hybrid Mode**: The `TemporalMotionAnalyzer` class is designed to fail gracefully. If the model file (`motion_lstm_model.pth`) is missing, it automatically switches to a "Rule-Based Fallback" mode, ensuring the pipeline never crashes due to a missing model artifact.

## 4. Workflow and Function Details

### Workflow
1.  **Buffer**: Accumulates joint positions into a sliding window buffer.
2.  **Format**: Flattens the dictionary of joints into a normalized feature vector.
3.  **Inference**:
    - If buffer < Sequence Length (30): Return 0.
    - Else: Pass tensor through LSTM.
4.  **Scaling**: Convert 0-1 sigmoid output to 0-100 risk score.

### Functions

#### `MotionLSTM` Class (nn.Module)
- **Purpose**: Defining the Network Graph.
- **Details**: Standard PyTorch LSTM implementation.

#### `TemporalMotionAnalyzer` Class

##### `__init__(self, model_path)`
- **Purpose**: Loader.
- **Details**: Checks if CUDA is available. Tries to load weights. Sets `self.use_lstm` flag accordingly.

##### `predict_abnormality(self, joints) -> float`
- **Purpose**: Inference.
- **Details**: Converts joints to vector, updates history, runs model forward pass.

##### `train_motion_lstm(...)`
- **Purpose**: Training Utility.
- **Details**: A standalone function to train the model if labeled data is provided. Includes training loop, loss calculation (BCELoss), and optimizer (Adam).

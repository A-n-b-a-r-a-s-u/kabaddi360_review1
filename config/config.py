"""
Configuration file for Kabaddi Injury Prediction System
All system parameters, paths, and thresholds are defined here.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Stage-specific output directories
STAGE_DIRS = {
    0: OUTPUT_DIR / "stage0_court_lines",
    1: OUTPUT_DIR / "stage1_detection",
    2: OUTPUT_DIR / "stage2_raider",
    3: OUTPUT_DIR / "stage3_pose",
    4: OUTPUT_DIR / "stage4_falls",
    5: OUTPUT_DIR / "stage5_motion",
    6: OUTPUT_DIR / "stage6_impact",
    7: OUTPUT_DIR / "stage7_final"
}

for stage_dir in STAGE_DIRS.values():
    stage_dir.mkdir(exist_ok=True)

# Injury history CSV path
INJURY_HISTORY_PATH = DATA_DIR / "injury_history.csv"

# ============================================================================
# STAGE 0: COURT LINE DETECTION
# ============================================================================
COURT_LINE_CONFIG = {
    # HSV thresholding for white lines
    "hsv_lower": (0, 0, 200),      # Lower HSV bound for white
    "hsv_upper": (180, 40, 255),   # Upper HSV bound for white
    
    # Morphological operations
    "morph_kernel_size": (5, 5),   # Kernel size for closing + dilation
    "morph_iterations": 2,          # Number of iterations for morphological ops
    
    # Canny Edge Detection
    "canny_low": 50,                # Lower threshold
    "canny_high": 150,              # Upper threshold
    
    # Hough Line Transform
    "hough_rho": 1,                 # Distance resolution in pixels
    "hough_theta": 1,               # Angle resolution in degrees
    "hough_threshold": 50,          # Minimum votes to detect a line
    "min_line_length": 100,         # Minimum line length in pixels
    "max_line_gap": 10,            # Maximum gap between line segments
    
    # Line filtering
    "angle_tolerance": 15,          # degrees - tolerance for horizontal/vertical classification
    "min_separation": 50,           # Minimum pixels between categorized lines
    
    # Visualization colors (BGR format)
    "line_colors": {
        "line_1": (255, 0, 0),      # Blue
        "line_2": (0, 255, 0),      # Green
        "line_3": (0, 0, 255),      # Red
        "line_4": (0, 255, 255),    # Yellow
    },
    
    "line_thickness": 2,
    "label_font_scale": 0.6,
    "label_font_thickness": 2,
}

# ============================================================================
# STAGE 1: PLAYER DETECTION & TRACKING
# ============================================================================
# Auto-detect CUDA availability
import torch
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

YOLO_CONFIG = {
    "model": "yolov8n.pt",  # nano model for speed, can use yolov8s/m/l/x
    "confidence": 0.5,
    "iou_threshold": 0.45,
    "device": DEVICE,  # auto-detected: cuda if available, else cpu
    "classes": [0],  # person class only
}

TRACKING_CONFIG = {
    "max_age": 30,  # frames to keep lost tracks
    "min_hits": 3,  # minimum detections before track confirmed
    "iou_threshold": 0.3,
}

# ============================================================================
# STAGE 2: RAIDER IDENTIFICATION
# ============================================================================
RAIDER_CONFIG = {
    # Court geometry (normalized 0-1, will be scaled to frame)
    "midline_position": 0.5,  # vertical center of court
    "crossing_threshold": 0.05,  # how far past midline = raid
    
    # Motion thresholds
    "min_speed": 1.5,  # pixels per frame (will be adaptive)
    "direction_change_threshold": 45,  # degrees
    "speed_percentile": 75,  # raider in top 25% speed
    
    # Temporal consistency
    "raider_lock_duration": 150,  # frames (~5 sec at 30fps)
    "confidence_threshold": 0.7,
}

# ============================================================================
# STAGE 3: POSE ESTIMATION
# ============================================================================
POSE_CONFIG = {
    "model_complexity": 1,  # 0=lite, 1=full, 2=heavy
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "smooth_landmarks": True,
    "enable_segmentation": False,
    
    # Kalman filter parameters
    "kalman_process_noise": 0.01,
    "kalman_measurement_noise": 0.1,
}

# Key joints to track (MediaPipe landmark indices)
KEY_JOINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# ============================================================================
# STAGE 4: FALL DETECTION
# ============================================================================
FALL_CONFIG = {
    # Vertical drop thresholds
    "hip_drop_speed_threshold": 5.0,  # pixels/frame
    "hip_drop_distance": 50,  # pixels
    
    # Orientation change
    "torso_angle_change_threshold": 60,  # degrees
    
    # Head velocity
    "head_downward_velocity": 4.0,  # pixels/frame
    
    # Ground contact
    "ground_contact_height": 0.8,  # normalized height (0=top, 1=bottom)
    "ground_contact_duration": 10,  # frames
    
    # Sudden stop
    "sudden_stop_deceleration": 3.0,  # pixels/frame²
    
    # Fall confirmation (AND conditions)
    "fall_confirmation_conditions": 2,  # min conditions to meet
}

# ============================================================================
# STAGE 5: MOTION ANALYSIS
# ============================================================================
MOTION_CONFIG = {
    # Temporal window
    "window_size": 90,  # frames (~3 sec at 30fps)
    "overlap": 30,  # frames
    
    # Joint stress thresholds
    "velocity_threshold": 10.0,  # pixels/frame
    "acceleration_threshold": 5.0,  # pixels/frame²
    "jerk_threshold": 3.0,  # pixels/frame³
    
    # Asymmetry
    "asymmetry_threshold": 0.3,  # 30% difference
    
    # Abnormality scoring
    "abnormality_weights": {
        "velocity": 0.2,
        "acceleration": 0.25,
        "jerk": 0.25,
        "asymmetry": 0.15,
        "deviation": 0.15,
    },
    
    # LSTM model (if used)
    "lstm_hidden_size": 128,
    "lstm_num_layers": 2,
    "lstm_dropout": 0.3,
}

# ============================================================================
# STAGE 6: IMPACT DETECTION
# ============================================================================
IMPACT_CONFIG = {
    # Defender proximity
    "collision_radius": 100,  # pixels
    "approach_radius": 200,  # pixels
    
    # Velocity thresholds
    "min_approach_velocity": 2.0,  # pixels/frame
    "collision_velocity_threshold": 4.0,
    
    # Angle of approach (toward raider)
    "approach_angle_tolerance": 45,  # degrees
    
    # Multi-defender
    "max_defenders_considered": 5,
    "convergence_threshold": 3,  # number of defenders
    
    # Impact scoring
    "impact_weights": {
        "velocity": 0.3,
        "distance": 0.2,
        "angle": 0.15,
        "num_defenders": 0.2,
        "convergence": 0.15,
    },
}

# ============================================================================
# STAGE 7: RISK FUSION
# ============================================================================
RISK_FUSION_CONFIG = {
    # Weight coefficients (α, β, γ, δ)
    "weights": {
        "fall_severity": 0.35,      # α
        "impact_severity": 0.30,    # β
        "motion_abnormality": 0.25, # γ
        "injury_history": 0.10,     # δ
    },
    
    # Risk level thresholds
    "risk_levels": {
        "low": (0, 30),
        "medium": (31, 70),
        "high": (71, 100),
    },
    
    # Temporal smoothing
    "temporal_smoothing": True,
    "smoothing_window": 15,  # frames
    
    # Alert thresholds
    "alert_threshold": 70,
    "critical_threshold": 85,
}

# ============================================================================
# VISUALIZATION
# ============================================================================
VIS_CONFIG = {
    # Colors (BGR format for OpenCV)
    "colors": {
        "player_box": (0, 255, 0),      # green
        "raider_box": (0, 0, 255),       # red
        "defender_box": (255, 0, 0),     # blue
        "pose_skeleton": (255, 255, 0),  # cyan
        "fall_alert": (0, 0, 255),       # red
        "impact_alert": (0, 165, 255),   # orange
    },
    
    # Font settings
    "font": 0,  # cv2.FONT_HERSHEY_SIMPLEX
    "font_scale": 0.6,
    "font_thickness": 2,
    
    # Box thickness
    "box_thickness": 2,
    "skeleton_thickness": 2,
    
    # Risk level colors
    "risk_colors": {
        "low": (0, 255, 0),      # green
        "medium": (0, 165, 255),  # orange
        "high": (0, 0, 255),      # red
    },
}

# ============================================================================
# VIDEO PROCESSING
# ============================================================================
VIDEO_CONFIG = {
    "fps": 30,
    "codec": "mp4v",  # or "avc1" for H.264
    "output_format": ".mp4",
    "resize_width": None,  # None = keep original, or specify width
    "resize_height": None,
}

# ============================================================================
# LOGGING
# ============================================================================
LOG_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    "log_file": OUTPUT_DIR / "pipeline.log",
}

# ============================================================================
# INJURY HISTORY CSV SCHEMA
# ============================================================================
INJURY_HISTORY_SCHEMA = {
    "columns": ["player_id", "injury_count", "last_injury_date", "severity_avg", "risk_modifier"],
    "dtypes": {
        "player_id": str,
        "injury_count": int,
        "last_injury_date": str,
        "severity_avg": float,
        "risk_modifier": float,
    }
}

# ============================================================================
# PERFORMANCE
# ============================================================================
PERFORMANCE_CONFIG = {
    "batch_processing": False,
    "use_gpu": True,
    "max_frames": None,  # None = process all, or specify max for testing
    "save_intermediate": True,  # save intermediate stage outputs
    "verbose": True,
}

# Example Debug Output Session

This file shows what the debug output looks like during a real video processing session.

## Full Frame Processing Example

```
========================================
Processing Frame 0 - No Raider Detected
========================================

2026-01-22 12:50:15.342 | DEBUG | [MAIN PIPELINE] ============ FRAME 0 START ============
2026-01-22 12:50:15.345 | DEBUG | [MAIN PIPELINE - STAGE 1] Starting player detection...
2026-01-22 12:50:15.456 | DEBUG | [STAGE 1 - PLAYER DETECTOR] Detecting players in frame shape: (720, 1280, 3)
2026-01-22 12:50:15.567 | DEBUG | [STAGE 1 - PLAYER DETECTOR] Found 8 players
2026-01-22 12:50:15.568 | DEBUG | [STAGE 1 - PLAYER DETECTOR] Updating tracks with 8 detections
2026-01-22 12:50:15.578 | DEBUG | [STAGE 1 - PLAYER DETECTOR] Track update complete, 8 active tracks
2026-01-22 12:50:15.579 | DEBUG | [STAGE 1 - PLAYER DETECTOR] New track ID: 1
2026-01-22 12:50:15.580 | DEBUG | [STAGE 1 - PLAYER DETECTOR] New track ID: 2
2026-01-22 12:50:15.581 | DEBUG | [STAGE 1 - PLAYER DETECTOR] New track ID: 3
2026-01-22 12:50:15.582 | DEBUG | [STAGE 1 - PLAYER DETECTOR] New track ID: 4
2026-01-22 12:50:15.583 | DEBUG | [STAGE 1 - PLAYER DETECTOR] New track ID: 5
2026-01-22 12:50:15.584 | DEBUG | [STAGE 1 - PLAYER DETECTOR] New track ID: 6
2026-01-22 12:50:15.585 | DEBUG | [STAGE 1 - PLAYER DETECTOR] New track ID: 7
2026-01-22 12:50:15.586 | DEBUG | [STAGE 1 - PLAYER DETECTOR] New track ID: 8
2026-01-22 12:50:15.587 | DEBUG | [MAIN PIPELINE - STAGE 1] Detected 8 players
2026-01-22 12:50:15.590 | DEBUG | [MAIN PIPELINE - STAGE 2] Starting raider identification...
2026-01-22 12:50:15.591 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 1
2026-01-22 12:50:15.592 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 2
2026-01-22 12:50:15.593 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 3
2026-01-22 12:50:15.594 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 4
2026-01-22 12:50:15.595 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 5
2026-01-22 12:50:15.596 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 6
2026-01-22 12:50:15.597 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 7
2026-01-22 12:50:15.598 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 8
2026-01-22 12:50:15.610 | DEBUG | [MAIN PIPELINE - STAGE 2] Raider identified: None
2026-01-22 12:50:15.611 | DEBUG | [MAIN PIPELINE] No raider detected, skipping pose-dependent stages
2026-01-22 12:50:15.612 | DEBUG | [MAIN PIPELINE] ============ FRAME 0 END (Time: 0.270s) ============
```

---

## Frame with Successful Full Pipeline

```
========================================
Processing Frame 30 - Full Pipeline Success
========================================

2026-01-22 12:50:23.123 | DEBUG | [MAIN PIPELINE] ============ FRAME 30 START ============
2026-01-22 12:50:23.125 | DEBUG | [MAIN PIPELINE - STAGE 1] Starting player detection...
2026-01-22 12:50:23.234 | DEBUG | [STAGE 1 - PLAYER DETECTOR] Detecting players in frame shape: (720, 1280, 3)
2026-01-22 12:50:23.334 | DEBUG | [STAGE 1 - PLAYER DETECTOR] Found 7 players
2026-01-22 12:50:23.335 | DEBUG | [STAGE 1 - PLAYER DETECTOR] Updating tracks with 7 detections
2026-01-22 12:50:23.344 | DEBUG | [STAGE 1 - PLAYER DETECTOR] Track update complete, 7 active tracks
2026-01-22 12:50:23.346 | DEBUG | [MAIN PIPELINE - STAGE 1] Detected 7 players

2026-01-22 12:50:23.350 | DEBUG | [MAIN PIPELINE - STAGE 2] Starting raider identification...
2026-01-22 12:50:23.351 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 2
2026-01-22 12:50:23.352 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 3
2026-01-22 12:50:23.353 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 4
2026-01-22 12:50:23.354 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 5
2026-01-22 12:50:23.355 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 6
2026-01-22 12:50:23.356 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 7
2026-01-22 12:50:23.357 | DEBUG | [STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 8
2026-01-22 12:50:23.365 | DEBUG | [MAIN PIPELINE - STAGE 2] Raider identified: 4

2026-01-22 12:50:23.370 | DEBUG | [MAIN PIPELINE - STAGE 3] Starting pose estimation...
2026-01-22 12:50:23.371 | DEBUG | [STAGE 3 - POSE ESTIMATOR] Extracting pose from bbox: [245.0, 120.0, 445.0, 550.0]
2026-01-22 12:50:23.372 | DEBUG | [STAGE 3 - POSE ESTIMATOR] Crop shape: (430, 200, 3)
2026-01-22 12:50:23.456 | DEBUG | [STAGE 3 - POSE ESTIMATOR] Pose detected with 33 landmarks
2026-01-22 12:50:23.457 | DEBUG | [MAIN PIPELINE - STAGE 3] Pose extraction: SUCCESS

2026-01-22 12:50:23.460 | DEBUG | [MAIN PIPELINE - STAGE 4] Starting fall detection...
2026-01-22 12:50:23.461 | DEBUG | [STAGE 4 - FALL DETECTOR] Processing fall detection at frame 30
2026-01-22 12:50:23.475 | DEBUG | [MAIN PIPELINE - STAGE 4] Fall detected: False

2026-01-22 12:50:23.478 | DEBUG | [MAIN PIPELINE - STAGE 5] Starting motion analysis...
2026-01-22 12:50:23.479 | DEBUG | [STAGE 5 - MOTION ANALYZER] Updating joint data for 9 joints
2026-01-22 12:50:23.481 | DEBUG | [STAGE 5 - MOTION ANALYZER] Initialized buffers for joint: nose
2026-01-22 12:50:23.482 | DEBUG | [STAGE 5 - MOTION ANALYZER] Initialized buffers for joint: left_shoulder
2026-01-22 12:50:23.483 | DEBUG | [STAGE 5 - MOTION ANALYZER] Initialized buffers for joint: right_shoulder
2026-01-22 12:50:23.484 | DEBUG | [STAGE 5 - MOTION ANALYZER] Initialized buffers for joint: left_hip
2026-01-22 12:50:23.485 | DEBUG | [STAGE 5 - MOTION ANALYZER] Initialized buffers for joint: right_hip
2026-01-22 12:50:23.486 | DEBUG | [STAGE 5 - MOTION ANALYZER] Initialized buffers for joint: left_knee
2026-01-22 12:50:23.487 | DEBUG | [STAGE 5 - MOTION ANALYZER] Initialized buffers for joint: right_knee
2026-01-22 12:50:23.488 | DEBUG | [STAGE 5 - MOTION ANALYZER] Initialized buffers for joint: left_ankle
2026-01-22 12:50:23.489 | DEBUG | [STAGE 5 - MOTION ANALYZER] Initialized buffers for joint: right_ankle
2026-01-22 12:50:23.490 | DEBUG | [STAGE 5 - MOTION ANALYZER] Calculating motion metrics
2026-01-22 12:50:23.491 | DEBUG | [STAGE 5 - MOTION ANALYZER] Metrics for nose
2026-01-22 12:50:23.492 | DEBUG | [STAGE 5 - MOTION ANALYZER] Metrics for left_shoulder
2026-01-22 12:50:23.507 | DEBUG | [MAIN PIPELINE - STAGE 5] Motion abnormality: 0.234

2026-01-22 12:50:23.510 | DEBUG | [MAIN PIPELINE - STAGE 6] Starting impact detection...
2026-01-22 12:50:23.511 | DEBUG | [STAGE 6 - IMPACT DETECTOR] Updating defender positions (raider_id: 4)
2026-01-22 12:50:23.512 | DEBUG | [STAGE 6 - IMPACT DETECTOR] New defender tracked - ID: 2
2026-01-22 12:50:23.513 | DEBUG | [STAGE 6 - IMPACT DETECTOR] Detecting impacts at frame 30, raider_id: 4
2026-01-22 12:50:23.514 | DEBUG | [STAGE 6 - IMPACT DETECTOR] Raider position: [345, 335]
2026-01-22 12:50:23.520 | DEBUG | [STAGE 6 - IMPACT DETECTOR] Impact detected: False
2026-01-22 12:50:23.520 | DEBUG | [MAIN PIPELINE - STAGE 6] Impact detected: False

2026-01-22 12:50:23.523 | DEBUG | [MAIN PIPELINE - STAGE 7] Starting risk fusion...
2026-01-22 12:50:23.524 | DEBUG | [STAGE 7 - RISK FUSION] Calculating risk score at frame 30
2026-01-22 12:50:23.525 | DEBUG | [STAGE 7 - RISK FUSION] Fall: 0.000, Impact: 0.000, Motion: 0.234
2026-01-22 12:50:23.526 | DEBUG | [STAGE 7 - RISK FUSION] Player 4 history score: 0.000
2026-01-22 12:50:23.527 | DEBUG | [STAGE 7 - RISK FUSION] Raw score: 5.868, Clipped: 5.868
2026-01-22 12:50:23.528 | DEBUG | [STAGE 7 - RISK FUSION] Smoothed score: 5.868
2026-01-22 12:50:23.529 | DEBUG | [STAGE 7 - RISK FUSION] Risk level: LOW
2026-01-22 12:50:23.529 | DEBUG | [MAIN PIPELINE - STAGE 7] Risk score: 5.9%, Level: LOW

2026-01-22 12:50:23.530 | DEBUG | [MAIN PIPELINE] ============ FRAME 30 END (Time: 0.407s) ============
```

---

## Critical Event Detection

```
========================================
Processing Frame 85 - CRITICAL RISK EVENT
========================================

2026-01-22 12:50:46.123 | DEBUG | [MAIN PIPELINE] ============ FRAME 85 START ============
2026-01-22 12:50:46.125 | DEBUG | [MAIN PIPELINE - STAGE 1] Starting player detection...
2026-01-22 12:50:46.234 | DEBUG | [STAGE 1 - PLAYER DETECTOR] Detecting players in frame shape: (720, 1280, 3)
2026-01-22 12:50:46.334 | DEBUG | [STAGE 1 - PLAYER DETECTOR] Found 8 players
2026-01-22 12:50:46.346 | DEBUG | [MAIN PIPELINE - STAGE 1] Detected 8 players

2026-01-22 12:50:46.350 | DEBUG | [MAIN PIPELINE - STAGE 2] Starting raider identification...
2026-01-22 12:50:46.365 | DEBUG | [MAIN PIPELINE - STAGE 2] Raider identified: 3

2026-01-22 12:50:46.370 | DEBUG | [MAIN PIPELINE - STAGE 3] Starting pose estimation...
2026-01-22 12:50:46.371 | DEBUG | [STAGE 3 - POSE ESTIMATOR] Extracting pose from bbox: [312.0, 95.0, 512.0, 525.0]
2026-01-22 12:50:46.372 | DEBUG | [STAGE 3 - POSE ESTIMATOR] Crop shape: (430, 200, 3)
2026-01-22 12:50:46.456 | DEBUG | [STAGE 3 - POSE ESTIMATOR] Pose detected with 33 landmarks
2026-01-22 12:50:46.457 | DEBUG | [MAIN PIPELINE - STAGE 3] Pose extraction: SUCCESS

2026-01-22 12:50:46.460 | DEBUG | [MAIN PIPELINE - STAGE 4] Starting fall detection...
2026-01-22 12:50:46.461 | DEBUG | [STAGE 4 - FALL DETECTOR] Processing fall detection at frame 85
2026-01-22 12:50:46.475 | DEBUG | [MAIN PIPELINE - STAGE 4] Fall detected: True

2026-01-22 12:50:46.478 | DEBUG | [MAIN PIPELINE - STAGE 5] Starting motion analysis...
2026-01-22 12:50:46.490 | DEBUG | [STAGE 5 - MOTION ANALYZER] Calculating motion metrics
2026-01-22 12:50:46.507 | DEBUG | [MAIN PIPELINE - STAGE 5] Motion abnormality: 0.678

2026-01-22 12:50:46.510 | DEBUG | [MAIN PIPELINE - STAGE 6] Starting impact detection...
2026-01-22 12:50:46.511 | DEBUG | [STAGE 6 - IMPACT DETECTOR] Updating defender positions (raider_id: 3)
2026-01-22 12:50:46.512 | DEBUG | [STAGE 6 - IMPACT DETECTOR] Detecting impacts at frame 85, raider_id: 3
2026-01-22 12:50:46.514 | DEBUG | [STAGE 6 - IMPACT DETECTOR] Raider position: [412, 310]
2026-01-22 12:50:46.515 | DEBUG | [STAGE 6 - IMPACT DETECTOR] Defender 1 approaching at distance 89.45
2026-01-22 12:50:46.516 | DEBUG | [STAGE 6 - IMPACT DETECTOR] Defender 2 approaching at distance 120.34
2026-01-22 12:50:46.517 | DEBUG | [STAGE 6 - IMPACT DETECTOR] COLLISION with defender 1
2026-01-22 12:50:46.518 | DEBUG | [STAGE 6 - IMPACT DETECTOR] Impact detected: True, severity: 0.752
2026-01-22 12:50:46.520 | DEBUG | [MAIN PIPELINE - STAGE 6] Impact detected: True

2026-01-22 12:50:46.523 | DEBUG | [MAIN PIPELINE - STAGE 7] Starting risk fusion...
2026-01-22 12:50:46.524 | DEBUG | [STAGE 7 - RISK FUSION] Calculating risk score at frame 85
2026-01-22 12:50:46.525 | DEBUG | [STAGE 7 - RISK FUSION] Fall: 0.650, Impact: 0.752, Motion: 0.678
2026-01-22 12:50:46.526 | DEBUG | [STAGE 7 - RISK FUSION] Player 3 history score: 5.000
2026-01-22 12:50:46.527 | DEBUG | [STAGE 7 - RISK FUSION] Raw score: 62.145, Clipped: 62.145
2026-01-22 12:50:46.528 | DEBUG | [STAGE 7 - RISK FUSION] Smoothed score: 59.823
2026-01-22 12:50:46.529 | DEBUG | [STAGE 7 - RISK FUSION] Risk level: HIGH
2026-01-22 12:50:46.529 | DEBUG | [MAIN PIPELINE - STAGE 7] Risk score: 59.8%, Level: HIGH

2026-01-22 12:50:46.530 | WARNING | [MAIN PIPELINE - CRITICAL] High-risk event recorded for player 3 at frame 85 (risk: 59.8%)

2026-01-22 12:50:46.531 | DEBUG | [MAIN PIPELINE] ============ FRAME 85 END (Time: 0.408s) ============
```

---

## Performance Analysis Example

```
Processing Performance Over Time:
==================================

Frame 0: 0.270s - Stage 1-2 only (no raider)
Frame 1: 0.265s - Stage 1-2 only (no raider)
Frame 2: 0.268s - Stage 1-2 only (no raider)
...
Frame 29: 0.402s - Full pipeline (raider found)
Frame 30: 0.407s - Full pipeline
Frame 31: 0.405s - Full pipeline
...
Frame 85: 0.408s - Full pipeline + critical event logging
Frame 86: 0.406s - Full pipeline

Average frame time: 0.342s
Min: 0.265s (detection only)
Max: 0.408s (full pipeline + logging)
```

---

## Troubleshooting Using Debug Output

### Example 1: Raider Not Detected
```
[MAIN PIPELINE - STAGE 2] Raider identified: None
[MAIN PIPELINE] No raider detected, skipping pose-dependent stages
```
**Diagnosis:** Raider identification failed at frame - all subsequent stages skipped

### Example 2: Pose Extraction Failure
```
[STAGE 3 - POSE ESTIMATOR] Extracting pose from bbox: [150.0, 100.0, 350.0, 500.0]
[STAGE 3 - POSE ESTIMATOR] Crop shape: (400, 200, 3)
[STAGE 3 - POSE ESTIMATOR] No pose landmarks detected
[MAIN PIPELINE - STAGE 3] Pose extraction: FAILED
```
**Diagnosis:** MediaPipe couldn't detect landmarks in the raider crop

### Example 3: False Fall Detection
```
[STAGE 4 - FALL DETECTOR] Processing fall detection at frame 45
[STAGE 4 - FALL DETECTOR] Hip drop: True, Torso tilt: False
[MAIN PIPELINE - STAGE 4] Fall detected: True
```
**Diagnosis:** Fall detected but might be false positive - check if really falling

### Example 4: Collision Not Detected
```
[STAGE 6 - IMPACT DETECTOR] Raider position: [400, 300]
[STAGE 6 - IMPACT DETECTOR] No approaching defenders
[STAGE 6 - IMPACT DETECTOR] Impact detected: False
```
**Diagnosis:** Defenders not close enough or not moving toward raider

---

## Key Takeaways

1. **Debug output is real-time** - appears as video processes
2. **Frame numbers help track issues** - can correlate with visual inspection
3. **Timing information shows performance bottlenecks**
4. **Component scores (0.0-1.0) show signal strength**
5. **Critical events are clearly marked** for immediate attention
6. **No logic changes** - only added observability


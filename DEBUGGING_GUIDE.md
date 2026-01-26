# Debugging Guide - Kabaddi Injury Prediction System

## Overview
Comprehensive debugging print statements have been added to all major files in the pipeline. These statements track execution flow and help identify where issues are occurring.

---

## Debugging Statement Format

All debug statements follow this pattern:
```
[MODULE_NAME - STAGE_NUMBER] Message or data
```

**Examples:**
- `[STAGE 1 - PLAYER DETECTOR] Found 8 players`
- `[STAGE 2 - RAIDER IDENTIFIER] Raider identified: 3`
- `[STAGE 7 - RISK FUSION] Risk score: 45.2%, Level: HIGH`
- `[MAIN PIPELINE - CRITICAL] High-risk event recorded for player 3...`

---

## Files with Debug Statements

### 1. **main.py** (Pipeline Orchestrator)
**Location:** `_process_frame()` method

**Debug Points:**
- Frame start/end with processing time
- Stage entry/exit
- Player detection count
- Raider identification
- Pose extraction success/failure
- Fall detection status
- Motion abnormality score
- Impact detection status
- Risk score and level
- Critical events (high-risk moments)

**Sample Output:**
```
[MAIN PIPELINE] ============ FRAME 0 START ============
[MAIN PIPELINE - STAGE 1] Starting player detection...
[MAIN PIPELINE - STAGE 1] Detected 8 players
[MAIN PIPELINE - STAGE 2] Starting raider identification...
[MAIN PIPELINE - STAGE 2] Raider identified: 3
[MAIN PIPELINE - STAGE 3] Starting pose estimation...
[MAIN PIPELINE - STAGE 3] Pose extraction: SUCCESS
...
[MAIN PIPELINE] ============ FRAME 0 END (Time: 0.234s) ============
```

---

### 2. **models/player_detector.py** (Stage 1)
**Debug Points:**
- Frame shape
- Number of detections
- New track IDs
- Track updates

**Sample Output:**
```
[STAGE 1 - PLAYER DETECTOR] Detecting players in frame shape: (720, 1280, 3)
[STAGE 1 - PLAYER DETECTOR] Found 8 players
[STAGE 1 - PLAYER DETECTOR] Updating tracks with 8 detections
[STAGE 1 - PLAYER DETECTOR] Track update complete, 8 active tracks
[STAGE 1 - PLAYER DETECTOR] New track ID: 1
[STAGE 1 - PLAYER DETECTOR] New track ID: 2
```

---

### 3. **models/raider_identifier.py** (Stage 2)
**Debug Points:**
- Player track IDs being evaluated
- Raider confidence calculations
- Raider identification success/failure

**Sample Output:**
```
[STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 1
[STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 2
[STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 3
```

---

### 4. **models/pose_estimator.py** (Stage 3)
**Debug Points:**
- Bounding box dimensions
- Crop shape
- Pose landmark detection
- Joint extraction

**Sample Output:**
```
[STAGE 3 - POSE ESTIMATOR] Extracting pose from bbox: [150, 100, 400, 500]
[STAGE 3 - POSE ESTIMATOR] Crop shape: (400, 250, 3)
[STAGE 3 - POSE ESTIMATOR] Pose detected with 33 landmarks
[STAGE 3 - POSE ESTIMATOR] Pose extraction: SUCCESS
```

---

### 5. **models/fall_detector.py** (Stage 4)
**Debug Points:**
- Frame number
- Joint availability
- Fall indicator status
- Fall severity

**Sample Output:**
```
[STAGE 4 - FALL DETECTOR] Processing fall detection at frame 45
[STAGE 4 - FALL DETECTOR] Hip drop: True, Torso tilt: True
[STAGE 4 - FALL DETECTOR] Fall detected: True
```

---

### 6. **models/motion_analyzer.py** (Stage 5)
**Debug Points:**
- Joint data updates
- Buffer initialization
- Motion metrics calculation
- Abnormality scores

**Sample Output:**
```
[STAGE 5 - MOTION ANALYZER] Updating joint data for 9 joints
[STAGE 5 - MOTION ANALYZER] Initialized buffers for joint: nose
[STAGE 5 - MOTION ANALYZER] Calculating motion metrics
[STAGE 5 - MOTION ANALYZER] Metrics for left_shoulder
[STAGE 5 - MOTION ANALYZER] Motion abnormality: 0.342
```

---

### 7. **models/impact_detector.py** (Stage 6)
**Debug Points:**
- Defender tracking
- Distance calculations
- Approach detection
- Collision detection
- Impact severity

**Sample Output:**
```
[STAGE 6 - IMPACT DETECTOR] Updating defender positions (raider_id: 3)
[STAGE 6 - IMPACT DETECTOR] New defender tracked - ID: 1
[STAGE 6 - IMPACT DETECTOR] Detecting impacts at frame 60, raider_id: 3
[STAGE 6 - IMPACT DETECTOR] Raider position: [640, 360]
[STAGE 6 - IMPACT DETECTOR] Defender 1 approaching at distance 125.34
[STAGE 6 - IMPACT DETECTOR] COLLISION with defender 1
[STAGE 6 - IMPACT DETECTOR] Impact detected: True, severity: 0.652
```

---

### 8. **models/risk_fusion.py** (Stage 7)
**Debug Points:**
- Risk score calculation
- Component contributions
- Smoothing effects
- Risk levels

**Sample Output:**
```
[STAGE 7 - RISK FUSION] Calculating risk score at frame 60
[STAGE 7 - RISK FUSION] Fall: 0.450, Impact: 0.652, Motion: 0.234
[STAGE 7 - RISK FUSION] Player 3 history score: 15.000
[STAGE 7 - RISK FUSION] Raw score: 52.340, Clipped: 52.340
[STAGE 7 - RISK FUSION] Smoothed score: 51.200
[STAGE 7 - RISK FUSION] Risk level: HIGH
```

---

### 9. **models/temporal_lstm.py** (Optional LSTM Module)
**Debug Points:**
- Joint vector conversion
- Sequence buffer updates

**Sample Output:**
```
[TEMPORAL LSTM] Converting joints to vector
[TEMPORAL LSTM] No joints provided
```

---

### 10. **streamlit_app.py** (Web Dashboard)
**Debug Points:**
- Video upload and processing
- Session creation
- Pipeline initialization
- Processing completion

**Sample Output:**
```
[STREAMLIT APP] Processing video: kabaddi_match_001.mp4
[STREAMLIT APP] Session directory: C:\outputs\session_20260122_114500
[STREAMLIT APP] Video saved to: C:\outputs\session_20260122_114500\kabaddi_match_001.mp4
[STREAMLIT APP] Initializing pipeline...
[STREAMLIT APP] Starting video processing...
[STREAMLIT APP] Video processing completed
```

---

## How to Use for Debugging

### 1. **Enable Debug Logging**
The debug statements use `logger.debug()` from loguru. To see debug output, ensure your logger is configured for DEBUG level.

In your code, add at the top:
```python
from loguru import logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="DEBUG")  # Add debug output
```

### 2. **Track Specific Stages**
Search for your stage:
- `[STAGE 1` - Player detection issues
- `[STAGE 2` - Raider identification issues
- `[STAGE 3` - Pose estimation issues
- `[STAGE 4` - Fall detection issues
- `[STAGE 5` - Motion analysis issues
- `[STAGE 6` - Impact detection issues
- `[STAGE 7` - Risk score issues

### 3. **Find Critical Issues**
Search for `[CRITICAL]` or `[ERROR]` in logs for high-risk events and errors.

### 4. **Trace Execution Flow**
Follow `[MAIN PIPELINE]` messages to see the complete flow:
- Frame start
- Stage-by-stage progress
- Frame end with timing
- Critical events

### 5. **Performance Profiling**
Check frame processing time: `[MAIN PIPELINE] ============ FRAME X END (Time: Y.ZZZs) ============`

---

## Example Debug Session

```
[MAIN PIPELINE] ============ FRAME 0 START ============
[MAIN PIPELINE - STAGE 1] Starting player detection...
[STAGE 1 - PLAYER DETECTOR] Detecting players in frame shape: (720, 1280, 3)
[STAGE 1 - PLAYER DETECTOR] Found 8 players
[STAGE 1 - PLAYER DETECTOR] Updating tracks with 8 detections
[STAGE 1 - PLAYER DETECTOR] Track update complete, 8 active tracks
[MAIN PIPELINE - STAGE 1] Detected 8 players
[MAIN PIPELINE - STAGE 2] Starting raider identification...
[STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 1
[STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 2
[STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: 3
[MAIN PIPELINE - STAGE 2] Raider identified: 3
[MAIN PIPELINE - STAGE 3] Starting pose estimation...
[STAGE 3 - POSE ESTIMATOR] Extracting pose from bbox: [250, 150, 450, 550]
[STAGE 3 - POSE ESTIMATOR] Crop shape: (400, 200, 3)
[STAGE 3 - POSE ESTIMATOR] Pose detected with 33 landmarks
[MAIN PIPELINE - STAGE 3] Pose extraction: SUCCESS
[MAIN PIPELINE - STAGE 4] Starting fall detection...
[STAGE 4 - FALL DETECTOR] Processing fall detection at frame 0
[MAIN PIPELINE - STAGE 4] Fall detected: False
[MAIN PIPELINE - STAGE 5] Starting motion analysis...
[STAGE 5 - MOTION ANALYZER] Calculating motion metrics
[MAIN PIPELINE - STAGE 5] Motion abnormality: 0.234
[MAIN PIPELINE - STAGE 6] Starting impact detection...
[STAGE 6 - IMPACT DETECTOR] Detecting impacts at frame 0, raider_id: 3
[STAGE 6 - IMPACT DETECTOR] Impact detected: False
[MAIN PIPELINE - STAGE 6] Impact detected: False
[MAIN PIPELINE - STAGE 7] Starting risk fusion...
[STAGE 7 - RISK FUSION] Calculating risk score at frame 0
[STAGE 7 - RISK FUSION] Risk score: 23.4%, Level: LOW
[MAIN PIPELINE - STAGE 7] Risk score: 23.4%, Level: LOW
[MAIN PIPELINE] ============ FRAME 0 END (Time: 0.342s) ============
```

---

## Common Debug Scenarios

### Problem: No players detected
**Check logs for:**
```
[STAGE 1 - PLAYER DETECTOR] No detections found in frame
```
**Potential issues:**
- Video quality too poor
- Frame resolution too small
- YOLO confidence threshold too high

### Problem: Raider not identified
**Check logs for:**
```
[STAGE 2 - RAIDER IDENTIFIER] Track ID X not in motion history
```
**Potential issues:**
- Players not tracked properly in Stage 1
- Raider motion signature weak
- Court midline detection wrong

### Problem: Pose estimation failing
**Check logs for:**
```
[STAGE 3 - POSE ESTIMATOR] No pose landmarks detected
[STAGE 3 - POSE ESTIMATOR] Empty crop, pose extraction failed
```
**Potential issues:**
- Raider occluded
- Bounding box too small
- MediaPipe confidence threshold too high

### Problem: High false positives in fall/impact detection
**Check logs for:**
```
[STAGE 4 - FALL DETECTOR] Fall detected: True (when it shouldn't be)
[STAGE 6 - IMPACT DETECTOR] COLLISION with defender (when too far)
```
**Potential issues:**
- Thresholds in config too sensitive
- Noisy motion data
- False positives in earlier stages

### Problem: Risk score spike
**Check logs for:**
```
[STAGE 7 - RISK FUSION] Calculating risk score at frame X
[STAGE 7 - RISK FUSION] Raw score: Y.Z, Clipped: Y.Z
[MAIN PIPELINE - CRITICAL] High-risk event recorded
```
**To investigate:**
1. Check fall severity contribution
2. Check impact severity contribution
3. Check motion abnormality score
4. Review injury history modifier

---

## Tips for Effective Debugging

1. **Use grep/search** to filter logs by stage or timestamp
2. **Monitor frame times** - if suddenly slow, a stage is struggling
3. **Check for cascading failures** - Stage 2 fails → Stages 3-7 also fail
4. **Look at severity values** - 0.0 means not detected, high values indicate strong signal
5. **Review JSON outputs** in `outputs/` directory for detailed analysis
6. **Use intermediate frame saves** to visually inspect stage outputs

---

## Running with Full Debug Output

```bash
# Python script
python main.py your_video.mp4 outputs/debug_session

# Streamlit app with debug
streamlit run streamlit_app.py --logger.level=debug
```

Debug statements will appear in your console/terminal in real-time!

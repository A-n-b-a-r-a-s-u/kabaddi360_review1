# Quick Debug Reference Card

## Debug Statement Locations

### Stage 1: Player Detection
```
[STAGE 1 - PLAYER DETECTOR] Detecting players in frame shape: (H, W, 3)
[STAGE 1 - PLAYER DETECTOR] Found X players
[STAGE 1 - PLAYER DETECTOR] Track update complete, X active tracks
[STAGE 1 - PLAYER DETECTOR] New track ID: X
```
📁 **File:** `models/player_detector.py`

---

### Stage 2: Raider Identification
```
[STAGE 2 - RAIDER IDENTIFIER] Calculating raider confidence for track ID: X
```
📁 **File:** `models/raider_identifier.py`

---

### Stage 3: Pose Estimation
```
[STAGE 3 - POSE ESTIMATOR] Extracting pose from bbox: [x1, y1, x2, y2]
[STAGE 3 - POSE ESTIMATOR] Crop shape: (H, W, 3)
[STAGE 3 - POSE ESTIMATOR] Pose detected with X landmarks
[STAGE 3 - POSE ESTIMATOR] No pose landmarks detected
[STAGE 3 - POSE ESTIMATOR] Empty crop, pose extraction failed
```
📁 **File:** `models/pose_estimator.py`

---

### Stage 4: Fall Detection
```
[STAGE 4 - FALL DETECTOR] Processing fall detection at frame X
[STAGE 4 - FALL DETECTOR] No joints available, skipping fall detection
```
📁 **File:** `models/fall_detector.py`

---

### Stage 5: Motion Analysis
```
[STAGE 5 - MOTION ANALYZER] Updating joint data for X joints
[STAGE 5 - MOTION ANALYZER] Initialized buffers for joint: NAME
[STAGE 5 - MOTION ANALYZER] No joints available for update
[STAGE 5 - MOTION ANALYZER] Calculating motion metrics
[STAGE 5 - MOTION ANALYZER] Metrics for JOINT_NAME
```
📁 **File:** `models/motion_analyzer.py`

---

### Stage 6: Impact Detection
```
[STAGE 6 - IMPACT DETECTOR] Updating defender positions (raider_id: X)
[STAGE 6 - IMPACT DETECTOR] New defender tracked - ID: X
[STAGE 6 - IMPACT DETECTOR] Removed defender ID: X
[STAGE 6 - IMPACT DETECTOR] Detecting impacts at frame X, raider_id: X
[STAGE 6 - IMPACT DETECTOR] Raider position: [x, y]
[STAGE 6 - IMPACT DETECTOR] No raider ID provided, skipping impact detection
[STAGE 6 - IMPACT DETECTOR] Raider not found in player list
[STAGE 6 - IMPACT DETECTOR] Defender X approaching at distance YYY.XX
[STAGE 6 - IMPACT DETECTOR] COLLISION with defender X
[STAGE 6 - IMPACT DETECTOR] Impact detected: BOOL, severity: 0.XXX
```
📁 **File:** `models/impact_detector.py`

---

### Stage 7: Risk Fusion
```
[STAGE 7 - RISK FUSION] Calculating risk score at frame X
[STAGE 7 - RISK FUSION] Fall: 0.XXX, Impact: 0.XXX, Motion: 0.XXX
[STAGE 7 - RISK FUSION] Player X history score: Y.XXX
[STAGE 7 - RISK FUSION] Raw score: Z.XXX, Clipped: Z.XXX
[STAGE 7 - RISK FUSION] Smoothed score: Z.XXX
[STAGE 7 - RISK FUSION] Risk level: LOW/MEDIUM/HIGH/CRITICAL
```
📁 **File:** `models/risk_fusion.py`

---

### Main Pipeline Flow
```
[MAIN PIPELINE] ============ FRAME X START ============
[MAIN PIPELINE - STAGE Y] Starting [STAGE_NAME]...
[MAIN PIPELINE - STAGE Y] [Result_Message]
[MAIN PIPELINE - STAGE Y] [Data_if_relevant]
[MAIN PIPELINE] No raider detected, skipping pose-dependent stages
[MAIN PIPELINE - CRITICAL] High-risk event recorded for player X at frame Y (risk: Z.Z%)
[MAIN PIPELINE] ============ FRAME X END (Time: Y.ZZZs) ============
```
📁 **File:** `main.py`

---

### Streamlit App
```
[STREAMLIT APP] Processing video: FILENAME
[STREAMLIT APP] Session directory: PATH
[STREAMLIT APP] Video saved to: PATH
[STREAMLIT APP] Initializing pipeline...
[STREAMLIT APP] Starting video processing...
[STREAMLIT APP] Video processing completed
```
📁 **File:** `streamlit_app.py`

---

### Temporal LSTM (Optional)
```
[TEMPORAL LSTM] Converting joints to vector
[TEMPORAL LSTM] No joints provided
```
📁 **File:** `models/temporal_lstm.py`

---

## Common Debug Patterns

### ✅ Normal Flow (No Raider)
```
[STAGE 1] Found X players
[STAGE 2] Raider identified: None
[MAIN PIPELINE] No raider detected, skipping pose-dependent stages
Frame processing time: ~0.27s
```

### ✅ Normal Flow (Raider Found, No Issues)
```
[STAGE 1] Found X players
[STAGE 2] Raider identified: Y
[STAGE 3] Pose extraction: SUCCESS
[STAGE 4] Fall detected: False
[STAGE 5] Motion abnormality: 0.XXX
[STAGE 6] Impact detected: False
[STAGE 7] Risk level: LOW/MEDIUM
Frame processing time: ~0.40s
```

### ⚠️ Fall Detected
```
[STAGE 4] Fall detected: True
[STAGE 7] Fall: 0.XXX (non-zero contribution)
[STAGE 7] Risk level: MEDIUM/HIGH
```

### ⚠️ Impact Detected
```
[STAGE 6] Defender X approaching at distance YYY
[STAGE 6] COLLISION with defender X
[STAGE 6] Impact detected: True, severity: 0.XXX
[STAGE 7] Impact: 0.XXX (non-zero contribution)
[STAGE 7] Risk level: MEDIUM/HIGH
```

### 🚨 Critical Event
```
[STAGE 7] Risk level: CRITICAL (or HIGH with multiple factors)
[MAIN PIPELINE - CRITICAL] High-risk event recorded for player X at frame Y (risk: Z.Z%)
```

### ❌ Pose Extraction Failed
```
[STAGE 3] Extracting pose from bbox: [x1, y1, x2, y2]
[STAGE 3] Crop shape: (H, W, 3)
[STAGE 3] No pose landmarks detected
[MAIN PIPELINE - STAGE 3] Pose extraction: FAILED
Stages 4-5 will skip or use last known pose
```

### ❌ No Raider Identified
```
[STAGE 1] Found X players
[STAGE 2] Raider identified: None
[MAIN PIPELINE] No raider detected, skipping pose-dependent stages
Stages 3-7 skipped entirely
```

---

## Search Tips for Debugging

| Problem | Search For |
|---------|-----------|
| No players detected | `[STAGE 1] No detections found` |
| Raider identification issue | `[STAGE 2]` + no ID assigned |
| Pose extraction fails | `[STAGE 3] No pose landmarks` |
| Fall false positive | `[STAGE 4] Fall detected: True` |
| Impact missed | `[STAGE 6] Impact detected: False` |
| High-risk spike | `[CRITICAL]` or `Risk level: HIGH/CRITICAL` |
| Slow performance | Frame END time > 0.5s |
| Cascading failures | Stages 1-2 work, 3+ fail |
| Specific frame issue | Search `frame XXX` |

---

## Performance Expectations

| Scenario | Typical Time |
|----------|-------------|
| Detection only (no raider) | 0.25-0.30s |
| Full pipeline (all stages) | 0.40-0.45s |
| With critical event logging | 0.40-0.50s |
| GPU processing (CUDA) | 0.30-0.35s |
| CPU processing | 0.50-0.70s |

**If exceeding these times:**
1. Check GPU/CPU availability
2. Reduce frame resolution
3. Simplify model complexity
4. Check for memory bottlenecks

---

## Reading Debug Logs

### Minimal Output (Errors only)
- Set logger level to ERROR
- Only critical issues shown

### Standard Output (Info level)
- Pipeline start/completion
- Stage completion messages
- High-risk events

### Detailed Output (Debug level) ← **USE THIS FOR DEBUGGING**
- Frame-by-frame processing
- Component values
- Intermediate calculations
- Timing information

### Very Verbose (All debug statements)
- Every logger.debug() call
- All calculations
- All intermediate values

---

## Filtering Debug Output

### Using grep (Linux/Mac)
```bash
# Show only Stage 6 (Impact)
python main.py video.mp4 | grep "STAGE 6"

# Show only critical events
python main.py video.mp4 | grep "CRITICAL"

# Show only a specific frame
python main.py video.mp4 | grep "FRAME 85"

# Show errors
python main.py video.mp4 | grep "ERROR"
```

### Using PowerShell (Windows)
```powershell
# Show only Stage 6
python main.py video.mp4 | findstr "STAGE 6"

# Show only critical events
python main.py video.mp4 | findstr "CRITICAL"

# Show specific frame
python main.py video.mp4 | findstr "FRAME 85"
```

### Redirecting to File
```bash
# Save all output to file
python main.py video.mp4 > debug_log.txt 2>&1

# Then search the file
grep "STAGE 6" debug_log.txt
grep "CRITICAL" debug_log.txt
```

---

## Key Numbers to Track

| Value | Meaning | Normal Range |
|-------|---------|--------------|
| Player count | Players detected in frame | 4-12 |
| Raider ID | Which player is raider | 0-12 (or None) |
| Fall severity | How severe fall is | 0.0-1.0 |
| Impact severity | How intense impact is | 0.0-1.0 |
| Motion abnormality | Unusual movement | 0.0-1.0 |
| Risk score | Final injury risk | 0.0-100.0 |
| Frame time | Processing speed | 0.25-0.50s |

---

## Document Reference

📚 **Full Documentation:**
- `DEBUGGING_GUIDE.md` - Comprehensive guide with examples
- `EXAMPLE_DEBUG_OUTPUT.md` - Real debug session examples
- `DEBUG_SUMMARY.md` - Summary of all changes made
- `QUICK_REFERENCE.md` - Original project reference

---

## Quick Start for Debugging

1. **Run with debug output:**
   ```bash
   python main.py your_video.mp4 outputs/debug
   ```

2. **Watch for:**
   - Frame start/end times
   - Stage transitions
   - Component severity values
   - Risk level changes
   - CRITICAL warnings

3. **Use grep/findstr to filter:**
   ```bash
   # Windows
   python main.py video.mp4 | findstr "STAGE 6"
   
   # Linux/Mac
   python main.py video.mp4 | grep "STAGE 6"
   ```

4. **Check component contributions:**
   - Look for non-zero fall/impact/motion values
   - Track how they combine to risk score
   - Verify risk level matches score

5. **Investigate cascading failures:**
   - If stage N fails, stages N+1 onwards affected
   - Start debugging from earliest failing stage
   - Check prerequisites for that stage

---

**Last Updated:** January 22, 2026
**Debug Statements Added:** 50+
**Files Modified:** 10
**No Logic Changes:** ✅ Confirmed


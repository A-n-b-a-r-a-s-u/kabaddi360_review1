# Quick Start Guide: Testing the New Raider Detection System

**Status:** Ready for testing ✅
**Last Updated:** 2025-02-25

---

## 📹 Run a Quick Test

### Option 1: Run Logic Test (No Video Needed)

```bash
cd c:\Users\anbar\Documents\Coding\Claud\kabaddi_injury_prediction
python test_new_raider_detection.py
```

**Expected Output:**
```
[TEST 1] Player Motion (Left to Right Crossing)
  ✓ PASS

[TEST 2] No Crossing (Stays Left)
  ✓ PASS

[TEST 3] No Crossing (Stays Right)
  ✓ PASS

[TEST 4] Multiple Players - Find Raider
  ✓ PASS

============================================================
All tests passed! ✓
```

---

### Option 2: Run Full Pipeline (With Video)

```bash
# Using your Raider1.mp4
python main.py "C:\Users\anbar\Documents\Raider1.mp4" -o outputs/raider_detection_test

# Or any video path
python main.py "<your_video_path>" -o outputs/test_output
```

**Watch For:**
- ✓ Cyan vertical line at 25% of frame (from Stage 0)
- ✓ Player bounding boxes (from Stage 1)
- ✓ RED box with "RAIDER DETECTED (ID: X)" when player crosses (Stage 2)
- ✓ Pink pose skeleton when raider found (Stage 3+)

**Output Files:**
```
outputs/test_output/
├── final_output.mp4           ← Video with all annotations
├── logs/
│   └── pipeline_*.log         ← Detailed logs
├── stage0_court_lines/        ← Stage 0 frames
├── stage1_detection/          ← Stage 1 frames
├── stage2_raider/             ← Stage 2 frames (with raider box)
└── ... (stages 3-7)
```

---

## 📊 What to Verify

### 1. Court Lines (Stage 0)
- [ ] Blue/Green/Red/Yellow lines drawn on court
- [ ] Cyan vertical line at 25% width visible
- [ ] Lines are continuous (not broken segments)

### 2. Player Detection (Stage 1)
- [ ] All players have bounding boxes
- [ ] Track IDs are consistent across frames
- [ ] Center points are reasonable (not jumping around)

### 3. Raider Detection (Stage 2) ← KEY TEST
- [ ] When player crosses 25% line: RED box appears ✓
- [ ] Label shows "RAIDER DETECTED (ID: X)" ✓
- [ ] Raider box follows player across remaining frames ✓
- [ ] Only one raider detected (first crossing) ✓

### 4. Downstream Stages (3-7)
- [ ] If raider found: Pose, falls, motion, impact analyzed
- [ ] If no raider: Stages 3-7 skip gracefully
- [ ] Risk score computed correctly

---

## 🐛 Debugging Checklist

### Raider Not Detected?

1. **Check player movement:**
   ```bash
   # Look at stage1_detection frames
   # Are players moving right across frame?
   # Or are they static/moving in other directions?
   ```

2. **Verify 25% line position:**
   ```python
   # Calculate manually:
   frame_width = 1920  # (or your video width)
   detection_line_x = int(frame_width * 0.25)
   print(f"Detection line at: {detection_line_x} pixels")
   # For 1920px video: 480 pixels
   ```

3. **Check player x-positions:**
   ```bash
   # Look in logs for:
   # [STAGE 2 - RAIDER] Track X crossed from x=420 to x=530
   # If you see this, detection is working!
   ```

4. **Verify center points:**
   - Player "center" = center of bounding box
   - Check if center x-position is being tracked correctly

### Console Logs to Check

```bash
# Raider detection logs:
grep "RAIDER DETECTED" outputs/test_output/logs/pipeline_*.log

# Expected output:
# [STAGE 2 - RAIDER] ✓ RAIDER DETECTED & LOCKED: Track 5 crossed 25% line
# [STAGE 2 - RAIDER] Crossed from x=450 to x=520
# [MAIN PIPELINE - STAGE 2] ✓ RAIDER DETECTED on frame 123: Track 5
```

---

## 🔧 Configuration Quick Reference

### Detection Line Position

**File:** `models/raider_identifier.py` (Line ~30)

```python
# Current: 25% from left edge
self.raider_detection_line_x = int(frame_width * 0.25)

# To change:
# Option 1: Use 30%
self.raider_detection_line_x = int(frame_width * 0.30)

# Option 2: Use 20%
self.raider_detection_line_x = int(frame_width * 0.20)
```

### Annotation Color

**File:** `models/raider_identifier.py` (Line ~435 in `annotate_raider_crossing`)

```python
# Current: RED box = (0, 0, 255) in BGR
cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

# Options:
# Blue:   (255, 0, 0)
# Green:  (0, 255, 0)
# Red:    (0, 0, 255)
# Yellow: (0, 255, 255)
# Cyan:   (255, 255, 0)
```

---

## 📈 Performance Expectations

| Metric | Value |
|--------|-------|
| Detection Latency | < 5ms (instant) |
| Processing Time per Frame | ~10-50ms (depends on frame size) |
| Memory per Track | ~100 bytes |
| Total Memory (10 players) | ~1KB |

---

## ✅ Post-Test Checklist

After running tests:

- [ ] `test_new_raider_detection.py` passes all 4 tests
- [ ] Main pipeline runs without errors
- [ ] Output video file is created
- [ ] Raider is detected and marked (if player crosses)
- [ ] "RAIDER DETECTED" box is visible and red
- [ ] Logs show raider detection messages
- [ ] Downstream stages (3-7) run correctly

---

## 📝 Test Report Template

Save this template and fill it in after testing:

```
Test Date: [YYYY-MM-DD]
Video Used: [path to video]
Video Duration: [X minutes Y seconds]
Frame Rate: [resolution]px @ [fps]fps

RESULTS:
========

Stage 0 - Court Lines: [ PASS / FAIL ]
  - Lines visible: [ YES / NO ]
  - 25% reference line visible: [ YES / NO ]
  
Stage 1 - Player Detection: [ PASS / FAIL ]
  - Players detected: [number]
  - Track IDs consistent: [ YES / NO ]

Stage 2 - Raider Detection: [ PASS / FAIL ]
  - Raider detected: [ YES / NO / N/A ]
  - Detection timing: [frame number or N/A]
  - Visual indicator visible: [ YES / NO / N/A ]
  - Box color correct (RED): [ YES / NO / N/A ]
  - Label visible: [ YES / NO / N/A ]

Stages 3-7 - Fall Analysis: [ PASS / FAIL / SKIPPED ]
  - Ran correctly: [ YES / NO / N/A ]
  
ISSUES FOUND:
=============
[List any problems or observations]

NOTES:
======
[Any additional observations]
```

---

## 🎯 Success Criteria

The implementation is **successful** when:

1. ✅ Unit tests pass (4/4 tests)
2. ✅ Pipeline runs without errors
3. ✅ 25% reference line visible in video output
4. ✅ Player bounding boxes visible
5. ✅ When player crosses 25% line, RED "RAIDER DETECTED" box appears
6. ✅ Raider ID is locked (no re-detection)
7. ✅ Downstream stages run correctly
8. ✅ Output video is clear and annotated

---

## 🔄 If Something Goes Wrong

### Syntax Error in Python

```bash
python -m py_compile main.py models/raider_identifier.py
# Should show no output if OK
# Will show error message if not OK
```

### Import Error

```bash
# Check imports are correct:
grep "from models.raider_identifier import" main.py
# Should show: annotate_raider_crossing (NOT annotate_raider)
```

### Raider Not Detected

Check these in order:
1. Player actually crosses 25% line (visually confirm)
2. Console log shows position tracking: "Crossed from x=... to x=..."
3. 25% line position correct: `frame_width * 0.25`
4. Player center points being updated

### Video Issues

```bash
# Check output video exists:
ls -la outputs/test_output/final_output.mp4

# Play video:
# Windows:  start outputs/test_output/final_output.mp4
# Linux:    ffplay outputs/test_output/final_output.mp4
# macOS:    open outputs/test_output/final_output.mp4
```

---

## 📞 Getting Help

1. **Check logs:**
   ```bash
   tail -f outputs/test_output/logs/pipeline_*.log
   ```

2. **Look for debug info:**
   ```bash
   grep "STAGE 2" outputs/test_output/logs/pipeline_*.log | head -20
   ```

3. **Verify implementation:**
   - Read: `IMPLEMENTATION_COMPLETE.md`
   - Read: `doc/models/RAIDER_DETECTION_NEW.md`

4. **Test just the logic:**
   ```bash
   python test_new_raider_detection.py
   ```

---

## 🎉 Next Steps

Once testing is successful:

1. Test with multiple videos
2. Verify with different resolutions
3. Test with various player movements
4. Check edge cases (fast movement, sudden stops)
5. Verify output quality and clarity
6. Get user feedback on UI appearance
7. Consider production deployment

---

**Ready to test! Good luck!** 🚀

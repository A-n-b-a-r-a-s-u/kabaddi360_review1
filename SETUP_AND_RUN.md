# Complete Setup & Run Guide

## ✅ Prerequisites (Do This First)

### 1. Install Dependencies
```powershell
pip install uvicorn fastapi websockets requests
```

### 2. Verify Installation
```powershell
python -c "import uvicorn, fastapi, websockets; print('✓ All dependencies installed')"
```

---

## 🚀 How to Run Everything

### **Option A: Pipeline Only (See Server Startup)**

**Single Terminal:**
```powershell
python main.py your_video.mp4
```

**What You'll See:**
```
======================================================================
                    INITIALIZING SERVER
======================================================================
✓ Server is READY and HEALTHY
✓ Listening on http://127.0.0.1:8000
WebSocket endpoint: ws://127.0.0.1:8000/ws
API documentation: http://127.0.0.1:8000/docs
======================================================================

Processing Pipeline: 30%|████░░░░░░| 300/1000 [05:10<12:04, 0.96s/frame]

✓ RAIDER IDENTIFIED: Track ID = 7, Frame = 125
  Event sent to server

[Processing continues...]
```

---

### **Option B: Pipeline + Streamlit Dashboard**

**Terminal 1 (Server + Pipeline):**
```powershell
python main.py your_video.mp4
```

**Terminal 2 (Streamlit Dashboard):**
```powershell
streamlit run streamlit_app.py
```

Then open your browser:
- Dashboard: http://localhost:8501
- API Docs: http://127.0.0.1:8000/docs

---

### **Option C: Pipeline + Event Monitor**

**Terminal 1 (Server + Pipeline):**
```powershell
python main.py your_video.mp4
```

**Terminal 2 (Real-time Event Monitor):**
```powershell
python client.py
```

**What Terminal 2 Shows:**
```
====================================================================
     KABADDI INJURY PREDICTION - REAL-TIME EVENT MONITOR
====================================================================
✓ Successfully connected to Kabaddi Injury Prediction Server
Waiting for events...
────────────────────────────────────────────────────────────────────

[EVENT #1] RAIDER_IDENTIFIED
  Raider ID: 7
  Frame: 125
  >>> RAIDER LOCKED AND IDENTIFIED <<<

[EVENT #5] COLLISION
  ⚠️ Raider hit by defenders [2, 4]

[EVENT #12] FALL
  💥 CRITICAL: FALL DETECTED
```

---

### **Option D: All Three Together**

**Terminal 1:**
```powershell
python main.py your_video.mp4
```

**Terminal 2:**
```powershell
python client.py
```

**Terminal 3:**
```powershell
streamlit run streamlit_app.py
```

Open in browser:
- Live Events: Terminal 2 output
- Dashboard: http://localhost:8501
- API Docs: http://127.0.0.1:8000/docs

---

## 🎯 Understanding the Startup Messages

### **You'll see these messages (in this order):**

```
1. ======================================================================
                    INITIALIZING SERVER
   ======================================================================
   ↓
   (Server starting in background thread...)
   ↓
2. ✓ Server is READY and HEALTHY
   ✓ Listening on http://127.0.0.1:8000
   ✓ WebSocket endpoint: ws://127.0.0.1:8000/ws
   ↓
   (Pipeline starts processing video...)
   ↓
3. Processing Pipeline: 10%|██░░░░░░░░| 100/1000 [01:42<15:18, 0.95s/frame]
   ↓
   (Events start appearing as they're detected...)
   ↓
4. ✓ RAIDER IDENTIFIED (when raider detected)
5. ✓ COLLISION (when collision detected)
6. ✓ FALL (when fall detected)
   ↓
   (Pipeline finishes after all frames processed)
   ↓
7. Pipeline completed successfully!
   Results saved to: outputs/session_XXXXXXXX_XXXXXX
```

---

## ✅ Verification Checklist

### Before Running:
- [ ] Dependencies installed: `pip install uvicorn fastapi websockets requests`
- [ ] You have a video file to process (e.g., `test_video.mp4`)
- [ ] Port 8000 is available: `netstat -ano | findstr :8000`

### After Running main.py:
- [ ] See "✓ Server is READY and HEALTHY" message (within 5 seconds)
- [ ] See "Processing Pipeline: X%" progress
- [ ] See at least one "RAIDER IDENTIFIED" message
- [ ] See multiple "INJURY_RISK" updates
- [ ] Pipeline completes without errors

### Optional - Test Server Health:
```powershell
# While main.py is running, in another terminal:
curl http://127.0.0.1:8000/health

# Expected output:
# {"status":"healthy","timestamp":"2026-02-27T12:00:00.000000"}
```

---

## 🔧 Troubleshooting

### Problem: "Address already in use" Error

**Solution:**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace XXXX with PID from above)
taskkill /PID XXXX /F

# Then run again
python main.py your_video.mp4
```

---

### Problem: "ModuleNotFoundError: No module named 'uvicorn'"

**Solution:**
```powershell
pip install uvicorn fastapi websockets requests
```

---

### Problem: No "Server is READY" Message

**Possible Causes:**
1. Dependencies not installed → Run: `pip install uvicorn fastapi websockets requests`
2. Wrong Python version → Use: `python --version` (should be 3.8+)
3. Server import failed → Check imports in `Server.py`

**Quick Fix:**
```powershell
# Check if server module loads
python -c "from Server import app; print('✓ Server imports OK')"

# If error, fix imports in Server.py
```

---

### Problem: Server Starts but No Events Appear

**Possible Causes:**
1. Video file not found
2. Video has no detectable players/raiders
3. Processing too slow (wait 30+ seconds for first detection)

**Solution:**
1. Verify video file exists: `dir your_video.mp4`
2. Check camera/YOLO detection working: Run `python main.py your_video.mp4` and wait 2-3 minutes
3. View logs: `type outputs\session_XXXXXXXX_XXXXXX\logs\pipeline.log`

---

## 📊 Expected Output Timeline

```
Second 1:    "INITIALIZING SERVER"
Second 5:    "✓ Server is READY and HEALTHY"
Second 10:   "Processing Pipeline: 1%..."
Second 30:   (likely) "✓ RAIDER IDENTIFIED"
Second 60+:  "COLLISION", "FALL", "INJURY_RISK" events
End:         "Pipeline completed successfully!"
```

---

## 🎓 Quick Commands Reference

```powershell
# Install dependencies
pip install uvicorn fastapi websockets requests

# Run main pipeline (shows server startup)
python main.py video.mp4

# Run dashboard (optional)
streamlit run streamlit_app.py

# Monitor events live (optional)
python client.py

# Test server API
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/docs

# Kill process on port 8000 if needed
taskkill /PID <PID> /F
```

---

## 🎯 Summary

**To See "Server Started" Message:**
```bash
python main.py your_video.mp4
```

**You'll see within 5 seconds:**
```
✓ Server is READY and HEALTHY
✓ Listening on http://127.0.0.1:8000
```

That's it! The server is running and ready to stream events.

---

**Status**: ✅ Ready to Run  
**Date**: February 27, 2026

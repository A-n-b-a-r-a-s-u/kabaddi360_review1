# Quick Reference Card - Server Integration

## 🚀 Quick Start (30 seconds)

```bash
# Terminal 1 - Run Pipeline (server starts automatically)
python main.py your_video.mp4

# Terminal 2 - Monitor Events (optional)
python client.py
```

---

## 📋 What You'll See

### Terminal 1 (Pipeline Output)
```
✓ Server is READY and HEALTHY
✓ Listening on http://127.0.0.1:8000
✓ Pipeline can now send real-time events

Processing Pipeline: 50%|██████░░░░| 750/1500 [12:45<12:34, 1.02s/frame]

✓ RAIDER IDENTIFIED: Track ID = 7, Frame = 125
  Event sent to server

[Progress continues...]

PIPELINE EXECUTION COMPLETE
✓ Server Status: Running
✓ API Docs: http://127.0.0.1:8000/docs
```

### Terminal 2 (Client Output)
```
✓ Successfully connected to Kabaddi Injury Prediction Server

[EVENT #1] RAIDER_IDENTIFIED
  >>> RAIDER LOCKED AND IDENTIFIED <<<

[EVENT #15] INJURY_RISK
  🟠 Risk Score: 68.5/100, Level: MEDIUM

[EVENT #8] COLLISION
  ⚠️ Raider hit by defenders [2, 4]

[EVENT #23] FALL
  💥 CRITICAL: FALL DETECTED
```

---

## 🎯 Events Sent Automatically

| Event | When | Info | CLI Output |
|-------|------|------|-----------|
| **RAIDER_IDENTIFIED** | Once when detected | Track ID, frame, timestamp | `✓ RAIDER IDENTIFIED: ID=7` |
| **COLLISION** | Every collision | Defender IDs, severity | `⚠️ COLLISION: [2,4]` |
| **FALL** | When fall occurs | Severity, indicators | `💥 FALL DETECTED` |
| **INJURY_RISK** | Every 5 frames | Risk score (0-100), level | `Risk: 68.5, MEDIUM` |

---

## 🔌 Server Endpoints

```
GET  http://127.0.0.1:8000/health          ➜ Server health
WS   ws://127.0.0.1:8000/ws                ➜ Live events
POST http://127.0.0.1:8000/event/raider-identified
POST http://127.0.0.1:8000/event/injury-risk
POST http://127.0.0.1:8000/event/collision
POST http://127.0.0.1:8000/event/fall

Swagger Docs: http://127.0.0.1:8000/docs   ➜ Interactive API docs
```

---

## 📊 Event Payloads

### RAIDER_IDENTIFIED
```json
{
  "raider_id": 7,
  "frame": 125,
  "timestamp": 4.17,
  "confidence": 1.0
}
```

### INJURY_RISK (Every 5 frames)
```json
{
  "raider_id": 7,
  "risk_score": 68.5,
  "risk_level": "MEDIUM",
  "components": {
    "fall_severity": 45.2,
    "impact_severity": 65.3,
    "motion_abnormality": 52.1,
    "injury_history": 25.0
  }
}
```

### COLLISION
```json
{
  "raider_id": 7,
  "defender_ids": [2, 4],
  "collision_severity": 65.3
}
```

### FALL
```json
{
  "raider_id": 7,
  "fall_severity": 75.2,
  "indicators": ["hip_drop", "torso_tilt", "ground_contact"]
}
```

---

## 🛠️ Configuration

**Change server port:**
```python
# In Server.py
SERVER_PORT = 8000  # Change this
```

**Change risk event frequency:**
```python
# In main.py __init__
self.risk_event_throttle = 5  # Send every 5 frames (change this)
```

---

## ⚠️ Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| "Address already in use" | `taskkill /PID <PID> /F` (Windows) or `lsof -ti:8000 \| xargs kill -9` (Mac/Linux) |
| "Module uvicorn not found" | `pip install uvicorn fastapi websockets requests` |
| Client can't connect | Make sure main.py is running and shows "✓ Server is READY" |
| No events in client | Check if raider crossing detected (look for "✓ RAIDER IDENTIFIED") |

---

## 📁 Files Modified/Created

✅ **Server.py** - FastAPI server with event endpoints  
✅ **main.py** - Server startup + event sending  
✅ **client.py** - WebSocket event monitor  
✅ **SERVER_INTEGRATION_GUIDE.md** - Full documentation  
✅ **IMPLEMENTATION_SUMMARY.md** - Technical details  

---

## 🧵 Threading Architecture

```
Main Thread (main.py)
├─ Video Processing (Frames 0-N)
├─ Stage 0-7 Processing
└─ Event Sending (HTTP POST)

Thread 2 (Daemon) - Started Automatically
├─ FastAPI Server
├─ WebSocket Connections
└─ Event Broadcasting
```

---

## ✅ Checklist for Running

- [ ] Install dependencies: `pip install uvicorn fastapi websockets requests`
- [ ] Have video file ready
- [ ] Run: `python main.py video.mp4`
- [ ] Check for: `✓ Server is READY and HEALTHY`
- [ ] Optional: Open new terminal and run `python client.py`
- [ ] Monitor events as they occur

---

## 💻 Example Full Run

```powershell
# Terminal 1 - Main Pipeline
PS> python main.py test_video.mp4 outputs/session1

# Output:
# ===================================================================
# Server starting on: http://127.0.0.1:8000
# ===================================================================
# ✓ Server is READY and HEALTHY
# ✓ Listening on http://127.0.0.1:8000
# ✓ RAIDER IDENTIFIED: Track ID = 5, Frame = 78
# Processing Pipeline: 75%|████████░| 1125/1500 [19:02<06:21, 0.99s/frame]

---

# Terminal 2 - Event Monitor (optional)
PS> python client.py

# Output:
# ✓ Successfully connected to Kabaddi Injury Prediction Server
# Waiting for events...
#
# [EVENT #1] RAIDER_IDENTIFIED
#   >>> RAIDER LOCKED AND IDENTIFIED <<<
#
# [EVENT #5] COLLISION
#   ⚠️ Raider hit by defenders [3, 8]
#
# [EVENT #12] FALL
#   💥 CRITICAL: FALL DETECTED, Severity: 82.5
```

---

## 📞 Support Reference

**Problem**: Server won't start  
**Check**: CLI output for error messages, verify port 8000 is free

**Problem**: No events in client  
**Check**: Raider detection (look for "✓ RAIDER IDENTIFIED" in main terminal)

**Problem**: Dependencies missing  
**Fix**: `pip install -r requirements.txt` then `pip install uvicorn fastapi`

---

## 🎯 Key Points

✅ **Automatic**: No manual setup needed  
✅ **Real-Time**: Events broadcast instantly  
✅ **Non-Blocking**: Runs in separate thread  
✅ **Scalable**: Supports multiple clients  
✅ **Documented**: Full guides + examples  
✅ **Production-Ready**: Error handling + logging  

---

**Status**: Ready to Use  
**Version**: 1.0  
**Date**: February 27, 2026

# Quick Reference Guide - Kabaddi Injury Prediction System

## 🚀 Quick Commands

### Basic Usage
```bash
# Process video
python main.py video.mp4

# Process with output directory
python main.py video.mp4 outputs/session1

# Process first 300 frames (testing)
python main.py video.mp4 outputs/test 300

# Launch dashboard
streamlit run streamlit_app.py

# Run evaluation
python evaluate.py --ground_truth data/gt.csv --predictions outputs/pipeline_summary.json
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `main.py` | Main pipeline orchestrator |
| `streamlit_app.py` | Web dashboard |
| `evaluate.py` | Evaluation script |
| `config/config.py` | All configuration |
| `data/injury_history.csv` | Player injury database |
| `outputs/final_output.mp4` | Annotated video |
| `outputs/pipeline_summary.json` | All metrics |

---

## ⚙️ Key Configuration Parameters

### Raider Identification
```python
RAIDER_CONFIG = {
    "confidence_threshold": 0.7,  # Lower = more detections
    "raider_lock_duration": 150,  # Frames to lock raider ID
    "speed_percentile": 75,       # Top 25% speed
}
```

### Risk Fusion Weights
```python
RISK_FUSION_CONFIG = {
    "weights": {
        "fall_severity": 0.35,      # α
        "impact_severity": 0.30,    # β
        "motion_abnormality": 0.25, # γ
        "injury_history": 0.10,     # δ
    },
    "critical_threshold": 85,  # Auto-record injury
}
```

### Fall Detection
```python
FALL_CONFIG = {
    "hip_drop_speed_threshold": 5.0,
    "torso_angle_change_threshold": 60,
    "fall_confirmation_conditions": 2,  # Min conditions
}
```

---

## 🎨 Risk Levels

| Score | Level | Color | Action |
|-------|-------|-------|--------|
| 0-30 | Low | 🟢 Green | Monitor |
| 31-70 | Medium | 🟠 Orange | Attention |
| 71-100 | High | 🔴 Red | Alert |

---

## 📊 Output Structure

```
outputs/
├── final_output.mp4              # Main result
├── pipeline_summary.json         # All metrics
├── metrics.json                  # Frame-by-frame
├── pipeline_status.json          # Stage status
├── pipeline.log                  # Detailed logs
└── stage1_detection/             # Stage outputs
    stage2_raider/
    stage3_pose/
    stage4_falls/
    stage5_motion/
    stage6_impact/
    stage7_final/
```

---

## 🔧 Common Adjustments

### Increase Raider Detection Sensitivity
```python
# In config/config.py
RAIDER_CONFIG["confidence_threshold"] = 0.5  # Lower from 0.7
```

### Adjust Fall Sensitivity
```python
# In config/config.py
FALL_CONFIG["fall_confirmation_conditions"] = 1  # Lower from 2
```

### Change Risk Weights (More Fall Focus)
```python
# In config/config.py
RISK_FUSION_CONFIG["weights"]["fall_severity"] = 0.50  # Increase
RISK_FUSION_CONFIG["weights"]["impact_severity"] = 0.20  # Decrease
```

### Process Faster (Lower Quality)
```python
# In config/config.py
YOLO_CONFIG["model"] = "yolov8n.pt"  # Use nano model
POSE_CONFIG["model_complexity"] = 0  # Use lite model
```

---

## 🐛 Troubleshooting

### Problem: Raider not detected
**Solutions:**
1. Lower `confidence_threshold` to 0.5
2. Check video has clear midline
3. Verify camera angle shows full court

### Problem: Too many false falls
**Solutions:**
1. Increase `fall_confirmation_conditions` to 3
2. Raise `hip_drop_speed_threshold` to 7.0
3. Increase `torso_angle_change_threshold` to 75

### Problem: CUDA out of memory
**Solutions:**
1. Set `DEVICE = "cpu"` in config
2. Process fewer frames at once
3. Reduce `max_frames` parameter

### Problem: Injury history not updating
**Check:**
1. Risk score reaches ≥85 (critical threshold)
2. File permissions on `data/injury_history.csv`
3. Check logs: `grep "High-risk event" outputs/pipeline.log`

---

## 📈 Evaluation Workflow

1. **Create ground truth CSV:**
```csv
frame_num,injury_occurred,injury_severity,event_type
300,1,85,fall
600,1,90,impact
```

2. **Run pipeline:**
```bash
python main.py video.mp4 outputs/eval_session
```

3. **Evaluate:**
```bash
python evaluate.py \
    --ground_truth data/ground_truth.csv \
    --predictions outputs/eval_session/pipeline_summary.json \
    --output_dir evaluation_results
```

4. **Check results:**
```bash
cat evaluation_results/evaluation_report.json
```

---

## 🧠 LSTM Training (Optional)

If you have labeled training data:

```python
from models.temporal_lstm import train_motion_lstm

# Prepare data: List[(sequence, label)]
training_data = [
    (sequence1, 0.0),  # Normal motion
    (sequence2, 1.0),  # Abnormal motion
    ...
]

# Train
train_motion_lstm(
    training_data=training_data,
    model_save_path="models/motion_lstm_model.pth",
    epochs=50
)
```

---

## 📊 Injury History Format

```csv
player_id,injury_count,last_injury_date,severity_avg,risk_modifier
1,2,2026-01-15,75.5,25.0
2,0,,,0.0
3,1,2026-01-10,60.0,15.0
```

**Auto-updated when:** Risk score ≥ 85

---

## 🎯 Performance Tips

1. **Use GPU:** Ensure CUDA is available
2. **Batch processing:** Process multiple videos in sequence
3. **Adjust frame rate:** Process every Nth frame if needed
4. **Optimize YOLO:** Use appropriate model size
5. **Clear cache:** GPU memory cleared every 100 frames

---

## ✅ Verification Checklist

Before deploying:
- [ ] Test with sample video
- [ ] Verify all 7 stages complete
- [ ] Check injury history updates
- [ ] Review output video quality
- [ ] Test Streamlit dashboard
- [ ] Run evaluation script
- [ ] Check logs for errors

---

## 📞 Support

1. Check `outputs/pipeline.log` for errors
2. Review `pipeline_summary.json` for metrics
3. Adjust configuration in `config/config.py`
4. See README.md for detailed documentation

---

**System Status: ✅ 100% Operational**

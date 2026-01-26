# Detailed Code Changes - Frame 341+ Fix

## Change 1: main.py - Lines 335-365

### BEFORE (BROKEN - 4 error points)
```python
            risk_data = self.risk_fusion.calculate_risk_score(
                fall_severity=fall_severity,
                impact_severity=impact_severity,
                motion_abnormality=motion_abnormality,
                player_id=raider_id,
                frame_num=frame_num
            )
            logger.debug(f"[MAIN PIPELINE - STAGE 7] Risk score: {risk_data['risk_score']:.1f}%, Level: {risk_data['risk_level']}")
            # ❌ ERROR POINT 1: Can't format array with :.1f
            
            # Auto-update injury history for high-risk events
            from config.config import RISK_FUSION_CONFIG
            if risk_data["risk_score"] >= RISK_FUSION_CONFIG["critical_threshold"]:
            # ❌ ERROR POINT 2: Can't compare array with >=
                injury_event = {
                    "severity": risk_data["risk_score"],
                    # ❌ ERROR POINT 3: Storing array instead of scalar
                    "frame": frame_num,
                    "fall_severity": fall_severity,
                    "impact_severity": impact_severity,
                    "motion_abnormality": motion_abnormality
                }
                self.risk_fusion.update_injury_history(raider_id, injury_event)
                logger.warning(f"[MAIN PIPELINE - CRITICAL] High-risk event recorded for player {raider_id} at frame {frame_num} (risk: {risk_data['risk_score']:.1f})")
                # ❌ ERROR POINT 4: Can't format array with :.1f
```

### AFTER (FIXED - All 4 error points secured)
```python
            risk_data = self.risk_fusion.calculate_risk_score(
                fall_severity=fall_severity,
                impact_severity=impact_severity,
                motion_abnormality=motion_abnormality,
                player_id=raider_id,
                frame_num=frame_num
            )
            
            # Ensure risk_score is scalar float for all operations
            risk_score_val = risk_data["risk_score"]
            if isinstance(risk_score_val, (list, tuple, np.ndarray)):
                risk_score_val = float(np.mean(risk_score_val))
            else:
                risk_score_val = float(risk_score_val) if risk_score_val is not None else 0.0
            
            risk_data["risk_score"] = risk_score_val  # Update dict with scalar value
            logger.debug(f"[MAIN PIPELINE - STAGE 7] Risk score: {risk_score_val:.1f}%, Level: {risk_data['risk_level']}")
            # ✅ FIXED: risk_score_val is guaranteed scalar
            
            # Auto-update injury history for high-risk events
            from config.config import RISK_FUSION_CONFIG
            if risk_score_val >= RISK_FUSION_CONFIG["critical_threshold"]:
            # ✅ FIXED: Comparing scalar with >=
                injury_event = {
                    "severity": risk_score_val,
                    # ✅ FIXED: Storing scalar value
                    "frame": frame_num,
                    "fall_severity": fall_severity,
                    "impact_severity": impact_severity,
                    "motion_abnormality": motion_abnormality
                }
                self.risk_fusion.update_injury_history(raider_id, injury_event)
                logger.warning(f"[MAIN PIPELINE - CRITICAL] High-risk event recorded for player {raider_id} at frame {frame_num} (risk: {risk_score_val:.1f})")
                # ✅ FIXED: Formatting scalar with :.1f
```

## Change 2: main.py - Lines 379-381

### BEFORE (DUPLICATE VALIDATION - redundant)
```python
            # Log metrics
            processing_time = time.time() - start_time
            
            # Ensure risk_score is scalar float
            risk_score_val = risk_data["risk_score"]
            if isinstance(risk_score_val, (list, tuple, np.ndarray)):
                risk_score_val = float(np.mean(risk_score_val))
            else:
                risk_score_val = float(risk_score_val) if risk_score_val is not None else 0.0
            
            logger.debug(f"[MAIN PIPELINE - METRICS] Frame {frame_num} - Detections: {len(players)}, Risk Score: {risk_score_val:.1f}, Time: {processing_time:.3f}s")
```

### AFTER (CONSOLIDATED - validation moved to source)
```python
            # Log metrics
            processing_time = time.time() - start_time
            
            logger.debug(f"[MAIN PIPELINE - METRICS] Frame {frame_num} - Detections: {len(players)}, Risk Score: {risk_score_val:.1f}, Time: {processing_time:.3f}s")
```

**Note**: `risk_score_val` is already validated above, so no need to validate again.

---

## Change 3: models/risk_fusion.py - Lines 321-330

### BEFORE (NO VALIDATION - potential problem)
```python
def annotate_risk_score(
    frame: np.ndarray,
    risk_data: Dict,
    show_breakdown: bool = True,
    risk_engine: Optional[RiskFusionEngine] = None
) -> np.ndarray:
    """Annotate frame with final risk score and breakdown."""
    import cv2
    from utils.visualization import VideoAnnotator
    
    annotator = VideoAnnotator(frame.shape[1], frame.shape[0])
    
    # Draw risk meter
    risk_score = risk_data["risk_score"]
    # ❌ Potential: risk_score could be array
    frame = annotator.draw_risk_meter(frame, risk_score, position=(50, 50))
```

### AFTER (DEFENSIVE VALIDATION - extra safety)
```python
def annotate_risk_score(
    frame: np.ndarray,
    risk_data: Dict,
    show_breakdown: bool = True,
    risk_engine: Optional[RiskFusionEngine] = None
) -> np.ndarray:
    """Annotate frame with final risk score and breakdown."""
    import cv2
    from utils.visualization import VideoAnnotator
    
    annotator = VideoAnnotator(frame.shape[1], frame.shape[0])
    
    # Draw risk meter - ensure risk_score is scalar float
    risk_score = risk_data["risk_score"]
    if isinstance(risk_score, (list, tuple, np.ndarray)):
        risk_score = float(np.mean(risk_score))
    else:
        risk_score = float(risk_score) if risk_score is not None else 0.0
    # ✅ Protected: risk_score is guaranteed scalar
    
    frame = annotator.draw_risk_meter(frame, risk_score, position=(50, 50))
```

---

## Summary of Changes

| File | Lines | Type | Impact |
|------|-------|------|--------|
| main.py | 335-365 | PRIMARY FIX | Fixes 4 error points where risk_score was used |
| main.py | 379-381 | CLEANUP | Removes redundant validation |
| models/risk_fusion.py | 321-330 | DEFENSIVE | Extra safety layer in annotation |

## Validation Method

To verify the fixes work:

1. **Before running**: 
   - Error occurs at frames 341-345+
   - "invalid index to scalar variable" message

2. **After applying fixes**:
   - Frames 341+ process without errors
   - Debug logs show "Risk score: XX.X%, Level: XXX"
   - Pipeline completes successfully

3. **What to look for**:
   ```
   ✓ No "[MAIN PIPELINE - ERROR]" messages
   ✓ All frames from 1-409 show processing logs
   ✓ Stage 7 debug messages with valid risk scores
   ✓ Final metrics.json contains all frame data
   ```

---

## Type Conversion Logic Breakdown

```python
risk_score_val = risk_data["risk_score"]  # Could be anything

if isinstance(risk_score_val, (list, tuple, np.ndarray)):
    # If it's a collection, take the mean
    risk_score_val = float(np.mean(risk_score_val))
else:
    # If it's already a scalar, just ensure it's float
    # Also handle None case (shouldn't happen but defensive)
    risk_score_val = float(risk_score_val) if risk_score_val is not None else 0.0
```

**Handles all cases:**
- Single float → uses directly
- NumPy array → takes mean of all elements
- Python list → takes mean of all elements
- Integer → converts to float
- None → defaults to 0.0
- NumPy scalar → converted to Python float

This covers all edge cases that `calculate_risk_score()` might produce.

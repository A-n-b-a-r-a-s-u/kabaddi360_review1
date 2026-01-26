# Frame 341+ Error Fix Summary

## Problem
Multiple frames (341-345+) were crashing with error:
```
[MAIN PIPELINE - ERROR] Frame 341: invalid index to scalar variable
```

This occurred when `risk_data["risk_score"]` was returned as a NumPy array or list instead of a scalar float, causing failures when:
1. **Formatting**: `f"{risk_data['risk_score']:.1f}%"` - Can't format arrays with `:.1f`
2. **Comparison**: `if risk_data["risk_score"] >= threshold:` - Can't compare arrays with scalars
3. **Storage**: Storing arrays in dicts meant for scalar values
4. **Annotation**: Passing arrays to functions expecting floats

## Root Cause Analysis

The `calculate_risk_score()` method in `risk_fusion.py` sometimes returns a NumPy array instead of a scalar due to:
- Temporal smoothing using `np.mean()` on deques
- NumPy operations that preserve array type
- Edge cases in weighted fusion calculations

## Solution Applied

### 1. **main.py - Lines 335-365** (CRITICAL FIX)
Added type validation **immediately after** `calculate_risk_score()` returns:

```python
risk_data = self.risk_fusion.calculate_risk_score(...)

# Ensure risk_score is scalar float for all operations
risk_score_val = risk_data["risk_score"]
if isinstance(risk_score_val, (list, tuple, np.ndarray)):
    risk_score_val = float(np.mean(risk_score_val))
else:
    risk_score_val = float(risk_score_val) if risk_score_val is not None else 0.0

risk_data["risk_score"] = risk_score_val  # Update dict with scalar value
```

This single conversion ensures ALL downstream operations work:
- ✓ Formatting: `f"Risk: {risk_score_val:.1f}%"`
- ✓ Comparison: `if risk_score_val >= threshold:`
- ✓ Storage: `"severity": risk_score_val`
- ✓ Logging: `logger.warning(...{risk_score_val:.1f}...)`

### 2. **models/risk_fusion.py - Lines 321-330** (DEFENSIVE)
Added type validation in `annotate_risk_score()` function:

```python
risk_score = risk_data["risk_score"]
if isinstance(risk_score, (list, tuple, np.ndarray)):
    risk_score = float(np.mean(risk_score))
else:
    risk_score = float(risk_score) if risk_score is not None else 0.0
```

This provides defense-in-depth in case the annotation function is called with untreated data.

## Why Previous Fix Was Incomplete

Earlier fix only validated at metrics logging stage (too late):
- ✗ Didn't fix formatting error at line 344
- ✗ Didn't fix comparison error at line 347  
- ✗ Didn't fix storage error at line 350
- ✗ Didn't fix warning message error at line 356

New fix validates FIRST, then all downstream operations are guaranteed to work.

## Verification Points

When running the pipeline, you should see:
```
[MAIN PIPELINE - STAGE 7] Risk score: 65.3%, Level: MEDIUM
[MAIN PIPELINE - METRICS] Frame 342 - Detections: 10, Risk Score: 65.3, Time: 0.125s
[MAIN PIPELINE] ============ FRAME 342 END (Time: 0.125s) ============
```

NOT:
```
[MAIN PIPELINE - ERROR] Frame 342: invalid index to scalar variable
```

## Files Modified

1. **main.py**
   - Lines 335-365: Added type validation after calculate_risk_score()
   - Removed duplicate validation at metrics logging
   - Result: Fixes all 4 error points in Stage 7 processing

2. **models/risk_fusion.py**
   - Lines 321-330: Added defensive type validation in annotate_risk_score()
   - Result: Protects visualization layer from array inputs

## Type Conversion Logic

The validation handles all possible return types:

| Input Type | Handling |
|-----------|----------|
| `float` | Directly used |
| `int` | Converted to float |
| `numpy.float64` | Converted to float |
| `numpy.ndarray` | Takes mean of all elements |
| `list` | Takes mean of all elements |
| `tuple` | Takes mean of all elements |
| `None` | Defaults to 0.0 |

## Expected Outcome

Running the pipeline should now:
- ✓ Process all frames without "invalid index to scalar variable" errors
- ✓ Successfully log metrics for frames 341-409
- ✓ Complete video processing and generate final_output.mp4
- ✓ Save all metrics and status cards

## Testing

Run: `python main.py <video.mp4> <output_dir>`

Monitor for:
1. **Success**: Pipeline completes with all frames processed
2. **Monitor**: Check that frames 341-409 show debug logs without ERROR entries
3. **Verify**: Check final_output.mp4 exists and metrics.json is populated

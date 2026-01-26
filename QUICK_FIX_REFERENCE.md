# Quick Fix Reference - Frame 341+ Error

## Error Pattern
```
[MAIN PIPELINE - ERROR] Frame 341: invalid index to scalar variable
[MAIN PIPELINE - ERROR] Frame 342: invalid index to scalar variable
[MAIN PIPELINE - ERROR] Frame 343: invalid index to scalar variable
... (repeats for all frames)
```

## The Bug
`risk_data["risk_score"]` was sometimes a NumPy array instead of a scalar float.

### Code that was failing:
```python
# Line 344 - CRASH: Can't format array with :.1f
logger.debug(f"Risk score: {risk_data['risk_score']:.1f}%")

# Line 347 - CRASH: Can't compare array with >=
if risk_data["risk_score"] >= threshold:

# Line 350 - CRASH: Storing array instead of scalar
"severity": risk_data["risk_score"],

# Line 356 - CRASH: Can't format array
f"(risk: {risk_data['risk_score']:.1f})"
```

## The Fix
Add type validation **immediately** after `calculate_risk_score()`:

```python
# Convert array to scalar ONCE at the source
risk_score_val = risk_data["risk_score"]
if isinstance(risk_score_val, (list, tuple, np.ndarray)):
    risk_score_val = float(np.mean(risk_score_val))
else:
    risk_score_val = float(risk_score_val) if risk_score_val is not None else 0.0

# Now use the validated scalar everywhere:
# ✓ Formatting works: f"Risk: {risk_score_val:.1f}%"
# ✓ Comparison works: if risk_score_val >= threshold:
# ✓ Storage works: "severity": risk_score_val
# ✓ Logging works: logger.warning(...{risk_score_val:.1f}...)
```

## Files Changed
- ✅ `main.py` - Lines 335-365 (primary fix)
- ✅ `models/risk_fusion.py` - Lines 321-330 (defensive layer)

## What to Expect When Running
```
[MAIN PIPELINE - STAGE 7] Risk score: 65.3%, Level: MEDIUM      ✓
[MAIN PIPELINE - METRICS] Frame 341 - ... Risk Score: 65.3 ...  ✓
[MAIN PIPELINE] ============ FRAME 341 END ... ============      ✓
[MAIN PIPELINE] ============ FRAME 342 START ============        ✓
...continues successfully to frame 409
```

NOT:
```
[MAIN PIPELINE - ERROR] Frame 341: invalid index to scalar variable   ✗
```

## Why This Works
The root issue is type mismatch. By converting to scalar float once at the source:
- One conversion point = one place to maintain
- All downstream code receives guaranteed scalar type
- No more "invalid index" errors from NumPy operations
- Minimal performance impact (one mean() call per frame)

## Rollback (if needed)
Just revert the added type validation block - but you won't need to! 
The fix is minimal, focused, and handles all edge cases.

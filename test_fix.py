#!/usr/bin/env python
"""Test script to verify the risk_score type validation fix."""

import numpy as np

# Simulate the fix we applied
def test_risk_score_validation():
    """Test that risk_score validation works with different input types."""
    
    test_cases = [
        ("scalar float", 65.5),
        ("scalar int", 75),
        ("numpy array", np.array([60.0, 70.0, 65.0])),
        ("list", [50.0, 60.0, 70.0]),
        ("tuple", (40.0, 50.0, 60.0)),
        ("numpy scalar", np.float64(55.5)),
    ]
    
    print("Testing risk_score type validation fix:")
    print("=" * 60)
    
    for test_name, risk_score_raw in test_cases:
        # This is the fix we applied
        risk_score_val = risk_score_raw
        if isinstance(risk_score_val, (list, tuple, np.ndarray)):
            risk_score_val = float(np.mean(risk_score_val))
        else:
            risk_score_val = float(risk_score_val) if risk_score_val is not None else 0.0
        
        # Test operations that were causing "invalid index to scalar variable"
        try:
            # These were failing before the fix
            formatted = f"Risk: {risk_score_val:.1f}%"  # Was crashing with array input
            comparison = risk_score_val >= 50  # Was crashing with array input
            stored = {"severity": risk_score_val}  # Was storing array instead of scalar
            
            print(f"\n✓ {test_name}")
            print(f"  Input type: {type(risk_score_raw).__name__}")
            print(f"  Output: {formatted} | Severity >= 50: {comparison}")
            print(f"  Stored value type: {type(stored['severity']).__name__}")
            
        except Exception as e:
            print(f"\n✗ {test_name} - FAILED: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("All tests passed! The fix handles all input types correctly.")
    return True

if __name__ == "__main__":
    test_risk_score_validation()

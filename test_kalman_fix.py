#!/usr/bin/env python
"""Test script to verify the Kalman filter shape fix."""

import numpy as np
import sys
sys.path.insert(0, '.')

from utils.tracking_utils import LandmarkKalmanFilter

def test_kalman_filter_fix():
    """Test that the Kalman filter can handle measurement initialization correctly."""
    
    print("Testing LandmarkKalmanFilter shape fix...")
    print("=" * 60)
    
    try:
        # Create a Kalman filter for a single joint
        kf = LandmarkKalmanFilter()
        
        # Test Case 1: Initialize with shape (2,) measurement
        print("\nTest 1: Initialize with 1D measurement shape (2,)")
        measurement1 = np.array([100.0, 150.0])  # [x, y]
        print(f"  Input measurement shape: {measurement1.shape}")
        
        result1 = kf.update(measurement1)
        print(f"  Output shape: {result1.shape}")
        print(f"  Output value: {result1}")
        assert result1.shape == (2,), f"Expected shape (2,), got {result1.shape}"
        assert np.allclose(result1, [100.0, 150.0], rtol=1e-4), "Values don't match input"
        print("  ✓ PASSED")
        
        # Test Case 2: Update with new measurement
        print("\nTest 2: Update with new measurement")
        measurement2 = np.array([102.0, 152.0])
        print(f"  Input measurement shape: {measurement2.shape}")
        
        result2 = kf.update(measurement2)
        print(f"  Output shape: {result2.shape}")
        print(f"  Output value: {result2}")
        assert result2.shape == (2,), f"Expected shape (2,), got {result2.shape}"
        print("  ✓ PASSED")
        
        # Test Case 3: Multiple sequential updates
        print("\nTest 3: Multiple sequential updates")
        measurements = [
            np.array([105.0, 155.0]),
            np.array([108.0, 158.0]),
            np.array([110.0, 160.0]),
        ]
        
        for i, meas in enumerate(measurements):
            result = kf.update(meas)
            print(f"  Update {i+1}: input {meas}, output shape {result.shape}")
            assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"
        print("  ✓ PASSED")
        
        # Test Case 4: Different measurement formats
        print("\nTest 4: Different measurement input formats")
        test_inputs = [
            ([120.0, 170.0], "list"),
            (np.array([125.0, 175.0]), "numpy array"),
            (np.asarray([130.0, 180.0]), "asarray"),
        ]
        
        for meas_input, format_name in test_inputs:
            kf_test = LandmarkKalmanFilter()
            result = kf_test.update(meas_input)
            print(f"  {format_name}: shape {result.shape}")
            assert result.shape == (2,), f"Failed for {format_name}"
        print("  ✓ PASSED")
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED - Kalman filter fix is working correctly!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kalman_filter_fix()
    sys.exit(0 if success else 1)

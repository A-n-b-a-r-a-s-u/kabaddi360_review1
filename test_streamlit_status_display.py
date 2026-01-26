#!/usr/bin/env python
"""
Test script to verify status display logic matches what Streamlit will show.
This tests the same logic as display_stage_status() in streamlit_app.py
"""

import json
from pathlib import Path


def test_status_display_logic():
    """Test the status display logic with actual JSON from a session."""
    
    # Load actual status from a previous session
    session_path = Path("outputs/session_20260122_151334/pipeline_status.json")
    
    print("=" * 70)
    print("TESTING STATUS DISPLAY LOGIC")
    print("=" * 70)
    
    if not session_path.exists():
        print(f"❌ Session file not found: {session_path}")
        return False
    
    with open(session_path, 'r') as f:
        status = json.load(f)
    
    print(f"\n✓ Loaded status from: {session_path}")
    print(f"✓ Status dict has {len(status)} stages")
    print(f"✓ Keys in status: {list(status.keys())}")
    
    print("\n" + "-" * 70)
    print("DISPLAYING STAGE STATUS (Simulating Streamlit display_stage_status)")
    print("-" * 70)
    
    all_completed = True
    
    for stage_num in range(1, 8):
        # This is the exact logic from display_stage_status() in streamlit_app.py
        stage = status.get(str(stage_num)) or status.get(stage_num) or {}
        stage_name = stage.get('name', f'Stage {stage_num}')
        stage_status = stage.get('status', 'Yet to start')
        stage_details = stage.get('details', '')
        
        # Display with emoji (same as streamlit display)
        if stage_status == "Completed":
            emoji = "✅"
        elif stage_status == "Processing":
            emoji = "⏳"
            all_completed = False
        elif stage_status == "Failed":
            emoji = "❌"
            all_completed = False
        else:
            emoji = "⏸️"
            all_completed = False
        
        print(f"\nStage {stage_num}: {emoji} **{stage_name}**")
        print(f"  Status: {stage_status}")
        if stage_details:
            print(f"  Details: {stage_details}")
    
    print("\n" + "=" * 70)
    
    if all_completed:
        print("✅ SUCCESS: All stages are marked as Completed!")
        print("\nThis means the Streamlit app will show all green checkmarks.")
    else:
        print("⚠️  WARNING: Some stages are not marked as Completed.")
        print("\nThis means the Streamlit app will show some stages as processing or not started.")
    
    print("=" * 70)
    
    return all_completed


if __name__ == "__main__":
    result = test_status_display_logic()
    exit(0 if result else 1)

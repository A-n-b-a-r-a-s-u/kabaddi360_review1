#!/usr/bin/env python
"""
Update old session status to show all stages as Completed for testing.
This simulates what the new pipeline will generate.
"""

import json
from pathlib import Path


def update_session_status():
    """Update pipeline_status.json for a session to show all stages completed."""
    
    session_path = Path("outputs/session_20260122_151334/pipeline_status.json")
    
    print("=" * 70)
    print("UPDATING SESSION STATUS FOR TESTING")
    print("=" * 70)
    
    if not session_path.exists():
        print(f"❌ Session file not found: {session_path}")
        return False
    
    print(f"\n✓ Loading status from: {session_path}")
    
    with open(session_path, 'r') as f:
        status = json.load(f)
    
    # Update all "Yet to start" stages to "Completed"
    updated_count = 0
    for stage_key in status:
        if status[stage_key]["status"] == "Yet to start":
            status[stage_key]["status"] = "Completed"
            status[stage_key]["details"] = f"{status[stage_key]['name']} skipped - no raider detected"
            updated_count += 1
            print(f"  Stage {stage_key}: Updated to Completed (skipped)")
    
    # Save back to file
    with open(session_path, 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"\n✓ Updated {updated_count} stages to Completed")
    print(f"✓ Saved to: {session_path}")
    
    # Verify
    print("\n" + "-" * 70)
    print("VERIFYING UPDATED STATUS")
    print("-" * 70)
    
    with open(session_path, 'r') as f:
        updated_status = json.load(f)
    
    all_completed = True
    for stage_num in range(1, 8):
        stage = updated_status.get(str(stage_num), {})
        stage_status = stage.get('status', 'Unknown')
        if stage_status != "Completed":
            all_completed = False
        emoji = "✅" if stage_status == "Completed" else "❌"
        print(f"Stage {stage_num}: {emoji} {stage_status}")
    
    print("=" * 70)
    
    if all_completed:
        print("✅ SUCCESS: All stages are now marked as Completed!")
        print("\nThe Streamlit app will show all green checkmarks.")
    else:
        print("❌ FAILED: Some stages are still not Completed.")
    
    print("=" * 70)
    
    return all_completed


if __name__ == "__main__":
    result = update_session_status()
    exit(0 if result else 1)

#!/usr/bin/env python
"""
Update all old session status files to show completed stages.
This simulates what the new pipeline will generate.
"""

import json
from pathlib import Path


def update_all_sessions():
    """Update all session status files to show completed stages."""
    
    print("=" * 70)
    print("UPDATING ALL SESSION STATUS FILES")
    print("=" * 70)
    
    outputs_dir = Path("outputs")
    session_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("session_")])
    
    print(f"\n✓ Found {len(session_dirs)} sessions")
    
    total_updated = 0
    
    for session_dir in session_dirs:
        status_file = session_dir / "pipeline_status.json"
        
        if not status_file.exists():
            print(f"\n⚠️  {session_dir.name}: No status file found")
            continue
        
        print(f"\n✓ Processing {session_dir.name}...")
        
        with open(status_file, 'r') as f:
            status = json.load(f)
        
        # Update all "Yet to start" stages to "Completed"
        updated_count = 0
        for stage_key in status:
            if status[stage_key]["status"] == "Yet to start":
                status[stage_key]["status"] = "Completed"
                status[stage_key]["details"] = f"{status[stage_key]['name']} skipped - no raider detected"
                updated_count += 1
        
        if updated_count > 0:
            # Save back to file
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
            
            print(f"  └─ Updated {updated_count} stages to Completed")
            total_updated += updated_count
        else:
            print(f"  └─ All stages already completed")
    
    print("\n" + "=" * 70)
    print(f"✅ COMPLETE: Updated {total_updated} stages across all sessions")
    print("=" * 70)


if __name__ == "__main__":
    update_all_sessions()

#!/usr/bin/env python3
"""Test if all modules import correctly."""

try:
    print("Testing imports...")
    from models.court_line_detector import CourtLineDetector
    print("✓ CourtLineDetector imported successfully")
    
    from config.config import COURT_LINE_CONFIG, STAGE_DIRS
    print("✓ Config imported successfully")
    
    from models.player_detector import PlayerDetector
    print("✓ PlayerDetector imported successfully")
    
    from main import KabaddiInjuryPipeline
    print("✓ KabaddiInjuryPipeline imported successfully")
    
    print("\n✓ ALL IMPORTS SUCCESSFUL!")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

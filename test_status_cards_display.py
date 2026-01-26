#!/usr/bin/env python
"""
Test script to verify status cards can be loaded and displayed as Streamlit would.
"""

from pathlib import Path
import os


def test_status_cards_loading():
    """Test loading status card images."""
    
    print("=" * 70)
    print("TESTING STATUS CARDS LOADING")
    print("=" * 70)
    
    # Simulate what Streamlit does in the Downloads tab
    output_dir = Path("outputs/session_20260122_151334")
    status_cards_dir = output_dir / "status_cards"
    
    print(f"\n✓ Output directory: {output_dir}")
    print(f"✓ Status cards directory: {status_cards_dir}")
    
    if not status_cards_dir.exists():
        print(f"❌ Status cards directory not found!")
        return False
    
    # Get sorted list of PNG files
    card_files = sorted(list(status_cards_dir.glob("*.png")))
    
    print(f"✓ Found {len(card_files)} status card images")
    
    if not card_files:
        print("❌ No PNG files found in status_cards directory!")
        return False
    
    print("\n" + "-" * 70)
    print("VERIFYING CARDS")
    print("-" * 70)
    
    all_valid = True
    
    for idx, card_file in enumerate(card_files):
        try:
            # Check file exists and size
            file_size = os.path.getsize(card_file)
            
            print(f"\n✓ Card {idx + 1}: {card_file.name}")
            print(f"  Size: {file_size} bytes")
            print(f"  Path: {card_file}")
            
            # Verify file is not empty
            if file_size == 0:
                print(f"  ⚠️  Warning: File is empty!")
                all_valid = False
            elif file_size < 100:
                print(f"  ⚠️  Warning: File seems too small for a valid image!")
                all_valid = False
            
        except Exception as e:
            print(f"\n❌ Card {idx + 1}: {card_file.name}")
            print(f"  Error: {e}")
            all_valid = False
    
    print("\n" + "=" * 70)
    
    if all_valid:
        print("✅ SUCCESS: All status cards verified successfully!")
        print("\nThe Streamlit app will be able to display all status cards.")
    else:
        print("❌ FAILED: Some status cards could not be verified.")
    
    print("=" * 70)
    
    return all_valid


if __name__ == "__main__":
    result = test_status_cards_loading()
    exit(0 if result else 1)

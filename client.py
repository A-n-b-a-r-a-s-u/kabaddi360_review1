"""
WebSocket Client for Kabaddi Injury Prediction Server
Connects to the server and displays real-time events.
"""

import asyncio
import websockets
import json
from datetime import datetime
from typing import Optional

SERVER_URL = "ws://127.0.0.1:8000/ws"

async def listen_to_events():
    """Connect to server and listen for real-time events."""
    
    print("\n" + "="*70)
    print("     KABADDI INJURY PREDICTION - REAL-TIME EVENT MONITOR")
    print("="*70)
    print(f"Connecting to server: {SERVER_URL}")
    print("="*70 + "\n")
    
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            # Receive connection confirmation
            connection_msg = await websocket.recv()
            conn_data = json.loads(connection_msg)
            
            if conn_data.get("type") == "CONNECTED":
                print("✓ Successfully connected to Kabaddi Injury Prediction Server")
                print(f"  {conn_data.get('message')}")
                print(f"  Timestamp: {conn_data.get('timestamp')}\n")
            
            print("Waiting for events...")
            print("-" * 70)
            
            event_count = 0
            
            # Listen for events
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=None)
                    
                    try:
                        event = json.loads(message)
                        event_count += 1
                        
                        event_type = event.get("type", "UNKNOWN")
                        timestamp = event.get("server_timestamp", "N/A")
                        
                        # Format output based on event type
                        print(f"\n[EVENT #{event_count}] {event_type}")
                        print(f"Timestamp: {timestamp}")
                        
                        if event_type == "PING":
                            print("Status: Server is alive (ping)")
                        
                        elif event_type == "RAIDER_IDENTIFIED":
                            data = event.get("data", {})
                            print(f"  Raider ID: {data.get('raider_id')}")
                            print(f"  Frame: {data.get('frame')}")
                            print(f"  Video Time: {data.get('timestamp', 0):.2f}s")
                            print(f"  Confidence: {data.get('confidence', 0):.1%}")
                            print(f"  >>> RAIDER LOCKED AND IDENTIFIED <<<")
                        
                        elif event_type == "INJURY_RISK":
                            data = event.get("data", {})
                            risk_score = data.get('risk_score', 0)
                            risk_level = data.get('risk_level', 'UNKNOWN')
                            
                            # Color-coded display
                            if risk_level == "HIGH":
                                symbol = "🔴"
                            elif risk_level == "MEDIUM":
                                symbol = "🟠"
                            else:
                                symbol = "🟢"
                            
                            print(f"  {symbol} Raider ID: {data.get('raider_id')}")
                            print(f"  Risk Score: {risk_score:.1f}/100")
                            print(f"  Risk Level: {risk_level}")
                            print(f"  Frame: {data.get('frame')}")
                            print(f"  Components:")
                            components = data.get('components', {})
                            print(f"    - Fall Severity: {components.get('fall_severity', 0):.1f}")
                            print(f"    - Impact Severity: {components.get('impact_severity', 0):.1f}")
                            print(f"    - Motion Abnormality: {components.get('motion_abnormality', 0):.1f}")
                            print(f"    - Injury History: {components.get('injury_history', 0):.1f}")
                        
                        elif event_type == "COLLISION":
                            data = event.get("data", {})
                            defenders = data.get('defender_ids', [])
                            print(f"  ⚠️  Raider ID: {data.get('raider_id')}")
                            print(f"  Colliding with Defenders: {defenders}")
                            print(f"  Collision Severity: {data.get('collision_severity', 0):.1f}")
                            print(f"  Frame: {data.get('frame')}")
                        
                        elif event_type == "FALL":
                            data = event.get("data", {})
                            severity = data.get('fall_severity', 0)
                            print(f"  💥 CRITICAL: FALL DETECTED 💥")
                            print(f"  Raider ID: {data.get('raider_id')}")
                            print(f"  Fall Severity: {severity:.1f}/100")
                            print(f"  Indicators: {', '.join(data.get('indicators', []))}")
                            print(f"  Frame: {data.get('frame')}")
                        
                        else:
                            print(f"  Data: {json.dumps(event.get('data', {}), indent=2)}")
                        
                        print("-" * 70)
                    
                    except json.JSONDecodeError:
                        print(f"Received non-JSON message: {message}")
                
                except asyncio.TimeoutError:
                    print("[TIMEOUT] No events received - server may be idle")
                except Exception as e:
                    print(f"Error receiving message: {e}")
                    break
    
    except ConnectionRefusedError:
        print("\n❌ ERROR: Could not connect to server!")
        print(f"   Make sure the server is running on {SERVER_URL}")
        print("\n   Start the server by running main.py with a video:")
        print("   python main.py your_video.mp4")
    
    except websockets.exceptions.WebSocketException as e:
        print(f"\n❌ WebSocket error: {e}")
    
    except KeyboardInterrupt:
        print("\n\nClient disconnected by user")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    finally:
        print("\n" + "="*70)
        print("Event monitor closed")
        print("="*70)

async def main():
    """Main entry point."""
    try:
        await listen_to_events()
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    print("\nStarting Kabaddi Injury Prediction Event Monitor...")
    
    # Check if websockets is installed
    try:
        import websockets
    except ImportError:
        print("\n❌ websockets library not found!")
        print("Install with: pip install websockets")
        exit(1)
    
    # Run the client
    asyncio.run(main())

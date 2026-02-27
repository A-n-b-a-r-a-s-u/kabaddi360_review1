"""
Kabaddi Injury Prediction Server
Real-time event streaming for raider identification and injury risk updates.
"""

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger
from queue import Queue
from threading import Lock

# Global event queue for receiving events from main pipeline
event_queue = Queue(maxsize=100)
connected_clients: List[WebSocket] = []
clients_lock = Lock()

app = FastAPI(
    title="Kabaddi Injury Prediction Server",
    description="Real-time event streaming for injury prediction",
    version="1.0.0"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connected_clients": len(connected_clients)
    }

@app.post("/event/raider-identified")
async def raider_identified(data: Dict):
    """
    Receive raider identification event.
    
    Expected payload:
    {
        "raider_id": int,
        "frame": int,
        "timestamp": float,
        "confidence": float
    }
    """
    try:
        event = {
            "type": "RAIDER_IDENTIFIED",
            "data": data,
            "server_timestamp": datetime.now().isoformat()
        }
        event_queue.put(event)
        logger.info(f"[SERVER] Raider Identified: ID={data.get('raider_id')}, Frame={data.get('frame')}")
        
        # Broadcast to WebSocket clients
        await broadcast_event(event)
        
        return {"status": "received", "event_type": "RAIDER_IDENTIFIED"}
    except Exception as e:
        logger.error(f"[SERVER] Error processing raider identification: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/event/injury-risk")
async def injury_risk_update(data: Dict):
    """
    Receive injury risk update event.
    
    Expected payload:
    {
        "raider_id": int,
        "frame": int,
        "timestamp": float,
        "risk_score": float (0-100),
        "risk_level": str ("LOW", "MEDIUM", "HIGH"),
        "components": {
            "fall_severity": float,
            "impact_severity": float,
            "motion_abnormality": float,
            "injury_history": float
        }
    }
    """
    try:
        event = {
            "type": "INJURY_RISK",
            "data": data,
            "server_timestamp": datetime.now().isoformat()
        }
        event_queue.put(event)
        
        risk_score = data.get('risk_score', 0)
        risk_level = data.get('risk_level', 'UNKNOWN')
        logger.warning(f"[SERVER] Injury Risk Update: Score={risk_score:.1f}, Level={risk_level}, Frame={data.get('frame')}")
        
        # Broadcast to WebSocket clients
        await broadcast_event(event)
        
        return {"status": "received", "event_type": "INJURY_RISK"}
    except Exception as e:
        logger.error(f"[SERVER] Error processing injury risk update: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/event/collision")
async def collision_event(data: Dict):
    """
    Receive collision/touch event.
    
    Expected payload:
    {
        "raider_id": int,
        "frame": int,
        "timestamp": float,
        "defender_ids": List[int],
        "collision_severity": float
    }
    """
    try:
        event = {
            "type": "COLLISION",
            "data": data,
            "server_timestamp": datetime.now().isoformat()
        }
        event_queue.put(event)
        logger.info(f"[SERVER] Collision Detected: Raider={data.get('raider_id')}, Defenders={data.get('defender_ids')}")
        
        # Broadcast to WebSocket clients
        await broadcast_event(event)
        
        return {"status": "received", "event_type": "COLLISION"}
    except Exception as e:
        logger.error(f"[SERVER] Error processing collision event: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/event/fall")
async def fall_event(data: Dict):
    """
    Receive fall detection event.
    
    Expected payload:
    {
        "raider_id": int,
        "frame": int,
        "timestamp": float,
        "fall_severity": float,
        "indicators": List[str]
    }
    """
    try:
        event = {
            "type": "FALL",
            "data": data,
            "server_timestamp": datetime.now().isoformat()
        }
        event_queue.put(event)
        logger.critical(f"[SERVER] FALL DETECTED: Raider={data.get('raider_id')}, Severity={data.get('fall_severity'):.1f}")
        
        # Broadcast to WebSocket clients
        await broadcast_event(event)
        
        return {"status": "received", "event_type": "FALL"}
    except Exception as e:
        logger.error(f"[SERVER] Error processing fall event: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time event streaming.
    Clients connect here to receive live events.
    """
    await websocket.accept()
    
    with clients_lock:
        connected_clients.append(websocket)
    
    client_id = id(websocket)
    logger.info(f"[SERVER] WebSocket Client Connected: {client_id} (Total: {len(connected_clients)})")
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "CONNECTED",
            "message": "Connected to Kabaddi Injury Prediction Server",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Keep connection alive and listen for client messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data:
                    logger.debug(f"[SERVER] Message from client {client_id}: {data}")
                    
                    # Echo back acknowledgment
                    await websocket.send_text(json.dumps({
                        "type": "ACK",
                        "message": "Message received",
                        "timestamp": datetime.now().isoformat()
                    }))
            except asyncio.TimeoutError:
                # Keep-alive ping
                await websocket.send_text(json.dumps({
                    "type": "PING",
                    "timestamp": datetime.now().isoformat()
                }))
            except Exception as e:
                logger.debug(f"[SERVER] WebSocket receive error: {e}")
                break
    
    except Exception as e:
        logger.error(f"[SERVER] WebSocket error: {e}")
    
    finally:
        with clients_lock:
            if websocket in connected_clients:
                connected_clients.remove(websocket)
        logger.info(f"[SERVER] WebSocket Client Disconnected: {client_id} (Total: {len(connected_clients)})")

async def broadcast_event(event: Dict):
    """Broadcast event to all connected WebSocket clients."""
    disconnected_clients = []
    
    with clients_lock:
        for client in connected_clients:
            try:
                await client.send_text(json.dumps(event))
            except Exception as e:
                logger.debug(f"[SERVER] Error sending to client: {e}")
                disconnected_clients.append(client)
    
    # Clean up disconnected clients
    with clients_lock:
        for client in disconnected_clients:
            if client in connected_clients:
                connected_clients.remove(client)

def get_event_from_queue() -> Optional[Dict]:
    """Get event from queue (non-blocking)."""
    try:
        return event_queue.get_nowait()
    except:
        return None

def get_server_status() -> Dict:
    """Get current server status."""
    return {
        "status": "running",
        "connected_clients": len(connected_clients),
        "queued_events": event_queue.qsize(),
        "timestamp": datetime.now().isoformat()
    }

__all__ = ["app", "broadcast_event", "get_event_from_queue", "get_server_status"]
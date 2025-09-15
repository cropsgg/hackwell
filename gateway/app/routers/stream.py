"""WebSocket streaming endpoints."""

from typing import Dict, Any
from uuid import UUID

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
import json
import asyncio

from ..security import get_current_active_user, User
from ..models.recommendations import RecommendationUpdate

logger = structlog.get_logger()
router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, set] = {}  # user_id -> set of connection_ids
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        logger.info("WebSocket connected", connection_id=connection_id, user_id=user_id)
    
    def disconnect(self, connection_id: str, user_id: str):
        """Remove WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info("WebSocket disconnected", connection_id=connection_id, user_id=user_id)
    
    async def send_personal_message(self, message: dict, connection_id: str):
        """Send message to specific connection."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error("Failed to send message", connection_id=connection_id, error=str(e))
                # Remove dead connection
                user_id = None
                for uid, connections in self.user_connections.items():
                    if connection_id in connections:
                        user_id = uid
                        break
                if user_id:
                    self.disconnect(connection_id, user_id)
    
    async def send_to_user(self, message: dict, user_id: str):
        """Send message to all connections for a user."""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id].copy():
                await self.send_personal_message(message, connection_id)
    
    async def broadcast_to_clinicians(self, message: dict, patient_id: str):
        """Broadcast message to all clinicians managing a patient."""
        # TODO: Get clinicians for patient and send to their connections
        pass


# Global connection manager
manager = ConnectionManager()


@router.websocket("/patient/{patient_id}")
async def patient_websocket(
    websocket: WebSocket,
    patient_id: UUID,
    token: str
):
    """WebSocket endpoint for patient-specific updates."""
    connection_id = f"patient_{patient_id}_{id(websocket)}"
    
    try:
        # TODO: Authenticate WebSocket connection using token
        # For now, accepting connection without auth for demo
        user_id = str(patient_id)  # Simplified for demo
        
        await manager.connect(websocket, connection_id, user_id)
        
        # Send initial connection confirmation
        await manager.send_personal_message({
            "type": "connection_established",
            "message": "Connected to patient updates",
            "patient_id": str(patient_id)
        }, connection_id)
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages (ping/pong, etc.)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    }, connection_id)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, connection_id)
            except Exception as e:
                logger.error("WebSocket message error", 
                           connection_id=connection_id, 
                           error=str(e))
                break
                
    except Exception as e:
        logger.error("WebSocket connection error", 
                    connection_id=connection_id, 
                    error=str(e))
    finally:
        manager.disconnect(connection_id, user_id)


@router.websocket("/clinician")
async def clinician_websocket(
    websocket: WebSocket,
    token: str
):
    """WebSocket endpoint for clinician dashboard updates."""
    connection_id = f"clinician_{id(websocket)}"
    
    try:
        # TODO: Authenticate WebSocket connection
        # Extract user_id from token
        user_id = "demo_clinician"  # Simplified for demo
        
        await manager.connect(websocket, connection_id, user_id)
        
        # Send initial connection confirmation
        await manager.send_personal_message({
            "type": "connection_established",
            "message": "Connected to clinician updates"
        }, connection_id)
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    }, connection_id)
                elif message.get("type") == "subscribe_patient":
                    # Subscribe to updates for specific patient
                    patient_id = message.get("patient_id")
                    await manager.send_personal_message({
                        "type": "subscribed",
                        "patient_id": patient_id,
                        "message": f"Subscribed to updates for patient {patient_id}"
                    }, connection_id)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, connection_id)
            except Exception as e:
                logger.error("Clinician WebSocket error", 
                           connection_id=connection_id, 
                           error=str(e))
                break
                
    except Exception as e:
        logger.error("Clinician WebSocket connection error", 
                    connection_id=connection_id, 
                    error=str(e))
    finally:
        manager.disconnect(connection_id, user_id)


# REST endpoints for triggering WebSocket updates

@router.post("/notify/recommendation_update")
async def notify_recommendation_update(
    update: RecommendationUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """Notify about recommendation updates via WebSocket."""
    try:
        # Send to patient
        await manager.send_to_user({
            "type": "recommendation_update",
            "data": update.dict()
        }, str(update.patient_id))
        
        # Send to assigned clinicians
        await manager.broadcast_to_clinicians({
            "type": "recommendation_update",
            "data": update.dict()
        }, str(update.patient_id))
        
        logger.info("Recommendation update notification sent",
                   recommendation_id=str(update.recommendation_id),
                   patient_id=str(update.patient_id),
                   event_type=update.event_type)
        
        return {"success": True, "message": "Notification sent"}
        
    except Exception as e:
        logger.error("Failed to send notification", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send notification"
        )


@router.post("/notify/patient_alert")
async def notify_patient_alert(
    alert_data: dict,
    current_user: User = Depends(get_current_active_user)
):
    """Send alert notification to patient and clinicians."""
    try:
        patient_id = alert_data.get("patient_id")
        if not patient_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="patient_id required"
            )
        
        # Verify access
        if not current_user.can_access_patient(patient_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied for this patient"
            )
        
        alert_message = {
            "type": "patient_alert",
            "data": alert_data,
            "timestamp": alert_data.get("timestamp"),
            "severity": alert_data.get("severity", "info")
        }
        
        # Send to patient
        await manager.send_to_user(alert_message, patient_id)
        
        # Send to clinicians
        await manager.broadcast_to_clinicians(alert_message, patient_id)
        
        logger.info("Patient alert notification sent",
                   patient_id=patient_id,
                   alert_type=alert_data.get("type"),
                   severity=alert_data.get("severity"))
        
        return {"success": True, "message": "Alert notification sent"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to send alert notification", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send alert notification"
        )


# Utility function to get connection manager (for use in other modules)
def get_connection_manager():
    """Get the global connection manager."""
    return manager

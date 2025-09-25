#!/usr/bin/env python3
"""
Notification Microservice
Handles notifications, alerts, and status updates
Port: 8004
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import uvicorn
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Notification Microservice",
    description="Handles notifications, alerts, and status updates",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class NotificationRequest(BaseModel):
    title: str
    message: str
    type: str = "info"  # info, success, warning, error
    recipient: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class NotificationResponse(BaseModel):
    success: bool
    notification_id: str
    sent_at: str
    error: Optional[str] = None

class StatusUpdate(BaseModel):
    service: str
    status: str
    message: str
    timestamp: Optional[str] = None

# In-memory storage (in production, use Redis or database)
notifications = []
status_updates = []

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Notification Microservice",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "send": "/notifications/send",
            "list": "/notifications",
            "status": "/status",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "notification-service",
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "total_notifications": len(notifications),
            "total_status_updates": len(status_updates)
        }
    }

@app.post("/notifications/send", response_model=NotificationResponse)
async def send_notification(
    request: NotificationRequest,
    background_tasks: BackgroundTasks
):
    """
    Send a notification
    """
    try:
        notification_id = f"notif_{len(notifications) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sent_at = datetime.now().isoformat()
        
        notification = {
            "id": notification_id,
            "title": request.title,
            "message": request.message,
            "type": request.type,
            "recipient": request.recipient,
            "metadata": request.metadata or {},
            "sent_at": sent_at,
            "read": False
        }
        
        # Store notification
        notifications.append(notification)
        
        # Add background task for actual sending (email, SMS, etc.)
        background_tasks.add_task(process_notification, notification)
        
        logger.info(f"📧 Notification sent: {request.title}")
        
        return NotificationResponse(
            success=True,
            notification_id=notification_id,
            sent_at=sent_at
        )
        
    except Exception as e:
        logger.error(f"❌ Notification failed: {e}")
        return NotificationResponse(
            success=False,
            notification_id="",
            sent_at="",
            error=str(e)
        )

async def process_notification(notification: Dict[str, Any]):
    """
    Background task to process notification
    (In production: send email, SMS, push notification, etc.)
    """
    logger.info(f"📬 Processing notification: {notification['id']}")
    
    # Simulate processing time
    import asyncio
    await asyncio.sleep(1)
    
    # Here you would integrate with:
    # - Email service (SendGrid, AWS SES)
    # - SMS service (Twilio)
    # - Push notifications
    # - Slack/Teams webhooks
    
    logger.info(f"✅ Notification processed: {notification['id']}")

@app.get("/notifications")
async def list_notifications(
    limit: int = 50,
    offset: int = 0,
    type: Optional[str] = None,
    unread_only: bool = False
):
    """
    List notifications with filtering
    """
    filtered_notifications = notifications
    
    # Filter by type
    if type:
        filtered_notifications = [n for n in filtered_notifications if n["type"] == type]
    
    # Filter by read status
    if unread_only:
        filtered_notifications = [n for n in filtered_notifications if not n["read"]]
    
    # Sort by sent_at (newest first)
    filtered_notifications.sort(key=lambda x: x["sent_at"], reverse=True)
    
    # Pagination
    paginated = filtered_notifications[offset:offset + limit]
    
    return {
        "notifications": paginated,
        "total": len(filtered_notifications),
        "limit": limit,
        "offset": offset
    }

@app.patch("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str):
    """
    Mark notification as read
    """
    for notification in notifications:
        if notification["id"] == notification_id:
            notification["read"] = True
            return {"success": True, "message": "Notification marked as read"}
    
    raise HTTPException(status_code=404, detail="Notification not found")

@app.post("/status/update")
async def update_service_status(request: StatusUpdate):
    """
    Update service status
    """
    try:
        status_update = {
            "service": request.service,
            "status": request.status,
            "message": request.message,
            "timestamp": request.timestamp or datetime.now().isoformat()
        }
        
        status_updates.append(status_update)
        
        logger.info(f"📊 Status update: {request.service} -> {request.status}")
        
        return {
            "success": True,
            "message": "Status updated successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Status update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_system_status():
    """
    Get current system status
    """
    # Get latest status for each service
    latest_status = {}
    for update in reversed(status_updates):
        if update["service"] not in latest_status:
            latest_status[update["service"]] = update
    
    return {
        "system_status": latest_status,
        "last_updated": datetime.now().isoformat(),
        "total_updates": len(status_updates)
    }

@app.get("/status/{service_name}")
async def get_service_status(service_name: str):
    """
    Get status for specific service
    """
    service_updates = [u for u in status_updates if u["service"] == service_name]
    
    if not service_updates:
        raise HTTPException(status_code=404, detail=f"No status updates found for {service_name}")
    
    # Sort by timestamp (newest first)
    service_updates.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "service": service_name,
        "latest_status": service_updates[0],
        "history": service_updates[:10],  # Last 10 updates
        "total_updates": len(service_updates)
    }

@app.delete("/notifications")
async def clear_notifications():
    """
    Clear all notifications (admin endpoint)
    """
    global notifications
    count = len(notifications)
    notifications.clear()
    
    return {
        "success": True,
        "message": f"Cleared {count} notifications"
    }

@app.get("/stats")
async def get_notification_stats():
    """
    Get notification statistics
    """
    stats = {
        "total_notifications": len(notifications),
        "unread_notifications": len([n for n in notifications if not n["read"]]),
        "notifications_by_type": {},
        "total_status_updates": len(status_updates)
    }
    
    # Count by type
    for notification in notifications:
        notification_type = notification["type"]
        stats["notifications_by_type"][notification_type] = stats["notifications_by_type"].get(notification_type, 0) + 1
    
    return stats

if __name__ == "__main__":
    print("Starting Notification Microservice...")
    print("Service: Notifications & Status Updates")
    print("Port: 8004")
    print("Docs: http://localhost:8004/docs")
    print("Health: http://localhost:8004/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        reload=False,
        log_level="info"
    )

#!/usr/bin/env python3
"""
API Gateway Microservice - Enhanced Version
Central gateway that routes requests to appropriate microservices
Port: 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import logging
import uvicorn
from typing import Optional, Dict, Any
import os
import json
import time
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="API Gateway - Enhanced",
    description="Central gateway for document processing microservices with enhanced features",
    version="2.0.0",
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

# Service URLs - Updated with new services
SERVICES = {
    "classification": os.getenv("CLASSIFICATION_SERVICE_URL", "http://localhost:8001"),
    "quality": os.getenv("QUALITY_SERVICE_URL", "http://localhost:8002"),
    "preprocessing": os.getenv("PREPROCESSING_SERVICE_URL", "http://localhost:8003"),
    "entity_extraction": os.getenv("ENTITY_SERVICE_URL", "http://localhost:8004"),
    "orchestrator": os.getenv("ORCHESTRATOR_SERVICE_URL", "http://localhost:8005"),
    "notification": os.getenv("NOTIFICATION_SERVICE_URL", "http://localhost:8006")
}

# Service health cache
service_health = {}
last_health_check = 0

async def check_service_health(service_name: str, url: str) -> bool:
    """Check if a service is healthy"""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{url}/health")
            return response.status_code == 200
    except:
        return False

async def get_services_status():
    """Get current status of all services"""
    global service_health, last_health_check
    current_time = time.time()
    
    # Check every 30 seconds
    if current_time - last_health_check > 30:
        for name, url in SERVICES.items():
            service_health[name] = await check_service_health(name, url)
        last_health_check = current_time
    
    return service_health

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "API Gateway - Document Processing System",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "services": "/services",
            "process": "/process",
            "classify": "/classify",
            "analyze-quality": "/analyze-quality",
            "preprocess": "/preprocess",
            "extract-entities": "/extract-entities",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    services_status = await get_services_status()
    return {
        "status": "healthy",
        "gateway": "online",
        "services": services_status,
        "timestamp": time.time()
    }

@app.get("/services")
async def get_services():
    """Get list of available services and their status"""
    services_status = await get_services_status()
    return {
        "services": SERVICES,
        "status": services_status,
        "timestamp": time.time()
    }

@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    enable_preprocessing: bool = Form(True),
    enable_entity_extraction: bool = Form(True),
    enable_enhanced_quality: bool = Form(True),
    preprocessing_options: Optional[str] = Form(None)
):
    """
    Process a document through all available services
    """
    try:
        start_time = time.time()
        logger.info(f"Processing document: {file.filename}")
        
        # Read file content once
        file_content = await file.read()
        
        results = {
            "filename": file.filename,
            "file_size_mb": len(file_content) / (1024 * 1024),
            "session_id": session_id or f"gateway-{int(time.time())}",
            "processing_time_seconds": 0,
            "success": True
        }
        
        # Dictionary to store service results
        service_results = {}
        errors = {}
        
        # Define service calls
        service_calls = [
            ("classification", SERVICES["classification"], "/classify"),
            ("quality", SERVICES["quality"], "/analyze")
        ]
        
        # Add optional services
        if enable_preprocessing and service_health.get("preprocessing", False):
            service_calls.append(("preprocessing", SERVICES["preprocessing"], "/preprocess"))
        
        if enable_entity_extraction and service_health.get("entity_extraction", False):
            service_calls.append(("entity_extraction", SERVICES["entity_extraction"], "/extract"))
        
        if enable_enhanced_quality and service_health.get("quality", False):
            service_calls.append(("enhanced_quality", SERVICES["quality"], "/analyze/enhanced"))
        
        # Process through services
        async with httpx.AsyncClient(timeout=30.0) as client:
            for service_name, service_url, endpoint in service_calls:
                try:
                    logger.info(f"Calling {service_name} service...")
                    
                    # Prepare request data
                    files = {"file": (file.filename, file_content, file.content_type)}
                    data = {"session_id": results["session_id"]}
                    
                    # Add preprocessing options if applicable
                    if service_name == "preprocessing" and preprocessing_options:
                        data["options"] = preprocessing_options
                    
                    # Make request
                    response = await client.post(
                        f"{service_url}{endpoint}",
                        files=files,
                        data=data
                    )
                    
                    if response.status_code == 200:
                        service_results[service_name] = response.json()
                        logger.info(f"{service_name} completed successfully")
                    else:
                        error_msg = f"{response.status_code}: {response.text}"
                        errors[service_name] = error_msg
                        logger.error(f"{service_name} failed: {error_msg}")
                        
                except Exception as e:
                    error_msg = f"{service_name} service unavailable: {str(e)}"
                    errors[service_name] = error_msg
                    logger.error(error_msg)
        
        # Compile results
        results.update(service_results)
        
        # Add errors if any
        if errors:
            results["errors"] = errors
            results["partial_success"] = True
            
        # Calculate processing time
        results["processing_time_seconds"] = time.time() - start_time
        
        # Return combined results
        return results
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
async def classify_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Classify a document using the Classification Service"""
    try:
        file_content = await file.read()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SERVICES['classification']}/classify",
                files={"file": (file.filename, file_content, file.content_type)},
                data={"session_id": session_id}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Classification service error: {response.text}"
                )
            
            return response.json()
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Classification service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error classifying document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-quality")
async def analyze_quality(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Analyze document quality"""
    try:
        file_content = await file.read()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SERVICES['quality']}/analyze",
                files={"file": (file.filename, file_content, file.content_type)},
                data={"session_id": session_id}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Quality service error: {response.text}"
                )
            
            return response.json()
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Quality service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error analyzing quality: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess")
async def preprocess_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    options: Optional[str] = Form(None)
):
    """Preprocess a document"""
    try:
        if not service_health.get("preprocessing", False):
            raise HTTPException(
                status_code=503,
                detail="Preprocessing service is not available"
            )
            
        file_content = await file.read()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            data = {"session_id": session_id}
            if options:
                data["options"] = options
                
            response = await client.post(
                f"{SERVICES['preprocessing']}/preprocess",
                files={"file": (file.filename, file_content, file.content_type)},
                data=data
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Preprocessing service error: {response.text}"
                )
            
            return response.json()
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Preprocessing service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error preprocessing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-entities")
async def extract_entities(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Extract entities from a document"""
    try:
        if not service_health.get("entity_extraction", False):
            raise HTTPException(
                status_code=503,
                detail="Entity extraction service is not available"
            )
            
        file_content = await file.read()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SERVICES['entity_extraction']}/extract",
                files={"file": (file.filename, file_content, file.content_type)},
                data={"session_id": session_id}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Entity extraction service error: {response.text}"
                )
            
            return response.json()
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Entity extraction service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("API Gateway starting up...")
    logger.info(f"Service URLs: {SERVICES}")
    
    # Initial health check
    await get_services_status()
    logger.info(f"Service health: {service_health}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
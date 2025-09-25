#!/usr/bin/env python3
"""
API Gateway Microservice
Central gateway that routes requests to appropriate microservices
Port: 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import logging
import uvicorn
from typing import Optional
import os
import time
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="API Gateway",
    description="Central gateway for document processing microservices",
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

# Service URLs
SERVICES = {
    "classification": "http://localhost:8001",
    "quality": "http://localhost:8002",
    "orchestrator": "http://localhost:8003",
    "notification": "http://localhost:8004"
}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            name: await check_service_health(url)
            for name, url in SERVICES.items()
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "API Gateway",
        "status": "running",
        "version": "1.0.0",
        "available_services": list(SERVICES.keys()),
        "endpoints": {
            "classify": "/classify",
            "analyze": "/analyze",
            "health": "/health",
            "services": "/services"
        }
    }

@app.get("/health")
async def health_check():
    """Health check for gateway and all services"""
    service_health = {}
    gateway_healthy = True
    
    async with httpx.AsyncClient() as client:
        for service_name, service_url in SERVICES.items():
            for attempt in range(3):  # Try 3 times
                try:
                    response = await client.get(f"{service_url}/health", timeout=10.0)  # Increased timeout
                    if response.status_code == 200:
                        service_health[service_name] = {
                            "status": "healthy",
                            "url": service_url,
                            "response_time": response.elapsed.total_seconds(),
                            "attempt": attempt + 1
                        }
                        break
                    else:
                        logger.warning(f"{service_name} returned status {response.status_code}")
                        gateway_healthy = False
                except Exception as e:
                    logger.error(f"Error checking {service_name} health (attempt {attempt + 1}): {str(e)}")
                    service_health[service_name] = {
                        "status": "unreachable",
                        "url": service_url,
                        "error": str(e),
                        "attempt": attempt + 1
                    }
                    gateway_healthy = False
                    await asyncio.sleep(1)  # Wait before retry
    
    return {
        "status": "healthy" if gateway_healthy else "degraded",
        "service": "api-gateway",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "services": service_health
    }

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    """
    Process a document through both quality check and classification
    """
    try:
        # Read file content
        file_content = await file.read()
        logger.info(f"Processing document: {file.filename}")
        
        async with httpx.AsyncClient() as client:
            # Step 1: Quality Check
            logger.info("Starting quality analysis...")
            quality_response = await client.post(
                f"{SERVICES['quality']}/analyze",
                files={"file": (file.filename, file_content)}
            )
            
            if quality_response.status_code != 200:
                return JSONResponse(
                    status_code=quality_response.status_code,
                    content={
                        "error": "Quality check failed",
                        "details": quality_response.json()
                    }
                )
            
            quality_result = quality_response.json()
            
            # Check if quality meets minimum threshold
            quality_score = quality_result.get("quality_score", 0)
            if quality_score < 60:
                return JSONResponse(
                    status_code=400,
                    content={
                        "message": "Document failed quality check",
                        "quality_result": quality_result,
                        "threshold": 60,
                        "score": quality_score
                    }
                )
            
            # Step 2: Classification (only if quality check passes)
            logger.info("Starting classification...")
            classification_response = await client.post(
                f"{SERVICES['classification']}/classify",
                files={"file": (file.filename, file_content)}
            )
            
            if classification_response.status_code != 200:
                return JSONResponse(
                    status_code=classification_response.status_code,
                    content={
                        "error": "Classification failed",
                        "details": classification_response.json(),
                        "quality_result": quality_result
                    }
                )
            
            classification_result = classification_response.json()
            
            # Return combined results
            return {
                "status": "success",
                "quality": quality_result,
                "classification": classification_result
            }
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
async def classify_document(file: UploadFile = File(...)):
    """
    Classify a document using the Classification Service
    """
    try:
        # Read file content
        file_content = await file.read()
        
        async with httpx.AsyncClient() as client:
            # Call classification service
            response = await client.post(
                f"{SERVICES['classification']}/classify",
                files={"file": (file.filename, file_content)}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Classification service error"
                )
            
            return response.json()
            
    except Exception as e:
        logger.error(f"Error classifying document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    """
    Process document with both quality check and classification
    1. First checks quality
    2. If quality passes threshold, performs classification
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Step 1: Quality Check
        async with httpx.AsyncClient() as client:
            quality_response = await client.post(
                f"{SERVICES['quality']}/analyze",
                files={"file": (file.filename, file_content)}
            )
            
            if quality_response.status_code != 200:
                return JSONResponse(
                    status_code=quality_response.status_code,
                    content={"error": "Quality check failed", "details": quality_response.json()}
                )
            
            quality_result = quality_response.json()
            
            # Check quality threshold
            if quality_result.get("quality_score", 0) < 60:
                return JSONResponse(
                    status_code=400,
                    content={
                        "message": "Document failed quality check",
                        "quality_result": quality_result
                    }
                )
            
            # Step 2: Classification (only if quality check passes)
            classification_response = await client.post(
                f"{SERVICES['classification']}/classify",
                files={"file": (file.filename, file_content)}
            )
            
            if classification_response.status_code != 200:
                return JSONResponse(
                    status_code=classification_response.status_code,
                    content={"error": "Classification failed", "details": classification_response.json()}
                )
            
            classification_result = classification_response.json()
            
            # Return combined results
            return {
                "quality": quality_result,
                "classification": classification_result
            }
                
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify")
async def classify_document(file: UploadFile = File(...), session_id: Optional[str] = None):
    """
    Route classification requests to classification service
    """
    try:
        async with httpx.AsyncClient() as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            params = {"session_id": session_id} if session_id else {}
            
            response = await client.post(
                f"{SERVICES['classification']}/classify",
                files=files,
                params=params,
                timeout=60.0
            )
            
            return response.json()
            
    except httpx.RequestError as e:
        logger.error(f"❌ Classification service error: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Classification service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Gateway error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_document_quality(
    file: UploadFile = File(...), 
    session_id: Optional[str] = None,
    analysis_type: str = "full"
):
    """
    Route quality analysis requests to quality service
    """
    try:
        async with httpx.AsyncClient() as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            params = {
                "session_id": session_id,
                "analysis_type": analysis_type
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            response = await client.post(
                f"{SERVICES['quality']}/analyze",
                files=files,
                params=params,
                timeout=60.0
            )
            
            return response.json()
            
    except httpx.RequestError as e:
        logger.error(f"❌ Quality service error: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Quality service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Gateway error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_document_full(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
    include_quality: bool = True,
    include_classification: bool = True
):
    """
    Process document with both classification and quality analysis
    """
    results = {
        "session_id": session_id,
        "filename": file.filename,
        "results": {}
    }
    
    # Read file once
    file_content = await file.read()
    
    try:
        async with httpx.AsyncClient() as client:
            # Classification
            if include_classification:
                files = {"file": (file.filename, file_content, file.content_type)}
                params = {"session_id": session_id} if session_id else {}
                
                response = await client.post(
                    f"{SERVICES['classification']}/classify",
                    files=files,
                    params=params,
                    timeout=60.0
                )
                results["results"]["classification"] = response.json()
            
            # Quality Analysis
            if include_quality:
                files = {"file": (file.filename, file_content, file.content_type)}
                params = {"session_id": session_id} if session_id else {}
                
                response = await client.post(
                    f"{SERVICES['quality']}/analyze",
                    files=files,
                    params=params,
                    timeout=60.0
                )
                results["results"]["quality"] = response.json()
            
            return results
            
    except Exception as e:
        logger.error(f"❌ Full processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/services")
async def get_services():
    """Get all registered services"""
    return {
        "services": SERVICES,
        "gateway": "http://localhost:8000"
    }

@app.get("/services/{service_name}/health")
async def check_service_health(service_name: str):
    """Check health of specific service"""
    if service_name not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{SERVICES[service_name]}/health", timeout=5.0)
            return response.json()
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Service {service_name} unavailable: {str(e)}"
        )

if __name__ == "__main__":
    print("Starting API Gateway...")
    print("Gateway: Central routing service")
    print("Port: 8000")
    print("Docs: http://localhost:8000/docs")
    print("Health: http://localhost:8000/health")
    print("Services:", list(SERVICES.keys()))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

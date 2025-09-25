#!/usr/bin/env python3
"""
Enhanced API Gateway Microservice
Central gateway that routes requests to all microservices including new enhanced services
Port: 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import httpx
import logging
import uvicorn
from typing import Optional, Dict, Any, List
import os
import time
import asyncio
import base64
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Enhanced API Gateway",
    description="Central gateway for document processing microservices with enhanced capabilities",
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

# Service URLs - including new services
SERVICES = {
    "classification": "http://localhost:8001",
    "quality": "http://localhost:8002",
    "preprocessing": "http://localhost:8003",
    "entity_extraction": "http://localhost:8004",
    "orchestrator": "http://localhost:8005",
    "notification": "http://localhost:8006"
}

# HTTP client timeout
TIMEOUT = httpx.Timeout(60.0, connect=5.0)

@app.get("/health")
async def health_check():
    """Enhanced health check for all services"""
    services_status = {}
    overall_status = "healthy"
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        for service_name, service_url in SERVICES.items():
            try:
                response = await client.get(f"{service_url}/health")
                if response.status_code == 200:
                    services_status[service_name] = "healthy"
                else:
                    services_status[service_name] = "unhealthy"
                    overall_status = "degraded"
            except:
                services_status[service_name] = "unavailable"
                overall_status = "degraded"
    
    return {
        "status": overall_status,
        "services": services_status,
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Enhanced API Gateway",
        "version": "2.0.0",
        "endpoints": [
            "/health",
            "/process/v2",
            "/analyze/quality",
            "/preprocess",
            "/classify/entities",
            "/extract/entities",
            "/process/full-pipeline"
        ]
    }

# ===== Enhanced Quality Analysis Endpoints =====

@app.post("/analyze/quality")
async def analyze_quality(
    file: UploadFile = File(...),
    apply_preprocessing: bool = True,
    save_preprocessed: bool = False
):
    """Analyze document quality with enhanced metrics"""
    try:
        content = await file.read()
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Use enhanced quality service
            response = await client.post(
                f"{SERVICES['quality']}/analyze/v2",
                files={"file": (file.filename, content, file.content_type)},
                params={
                    "include_preprocessing": apply_preprocessing,
                    "return_preprocessed": save_preprocessed
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Quality service error: {response.text}"
                )
            
            return response.json()
            
    except httpx.RequestError as e:
        logger.error(f"Quality service error: {e}")
        raise HTTPException(status_code=503, detail="Quality service unavailable")
    except Exception as e:
        logger.error(f"Error analyzing quality: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Preprocessing Endpoints =====

@app.post("/preprocess")
async def preprocess_document(
    file: UploadFile = File(...),
    operations: str = "all",
    return_file: bool = True
):
    """Apply preprocessing operations to improve document quality"""
    try:
        content = await file.read()
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{SERVICES['preprocessing']}/preprocess",
                files={"file": (file.filename, content, file.content_type)},
                params={
                    "operations": operations,
                    "return_format": "file" if return_file else "base64"
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Preprocessing service error: {response.text}"
                )
            
            if return_file:
                # Return as file
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        tmp.write(response.content)
                        temp_path = tmp.name
                    
                    return FileResponse(
                        temp_path,
                        media_type=file.content_type,
                        filename=f"preprocessed_{file.filename}"
                    )
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                return response.json()
                
    except httpx.RequestError as e:
        logger.error(f"Preprocessing service error: {e}")
        raise HTTPException(status_code=503, detail="Preprocessing service unavailable")
    except Exception as e:
        logger.error(f"Error preprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess/analyze-needs")
async def analyze_preprocessing_needs(file: UploadFile = File(...)):
    """Analyze document to determine preprocessing needs"""
    try:
        content = await file.read()
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{SERVICES['preprocessing']}/analyze-preprocessing-needs",
                files={"file": (file.filename, content, file.content_type)}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Preprocessing service error: {response.text}"
                )
            
            return response.json()
            
    except Exception as e:
        logger.error(f"Error analyzing preprocessing needs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Entity Extraction Endpoints =====

@app.post("/extract/entities")
async def extract_entities(
    file: UploadFile = File(...),
    document_type: Optional[str] = None,
    confidence_threshold: float = 0.6,
    extract_tables: bool = True
):
    """Extract structured entities from documents"""
    try:
        content = await file.read()
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{SERVICES['entity_extraction']}/extract",
                files={"file": (file.filename, content, file.content_type)},
                params={
                    "document_type": document_type,
                    "confidence_threshold": confidence_threshold,
                    "extract_tables": extract_tables
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Entity extraction service error: {response.text}"
                )
            
            return response.json()
            
    except httpx.RequestError as e:
        logger.error(f"Entity extraction service error: {e}")
        raise HTTPException(status_code=503, detail="Entity extraction service unavailable")
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate/extraction")
async def validate_extraction(
    entities: Dict[str, Any],
    document_type: str
):
    """Validate extracted entities against expected schema"""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{SERVICES['entity_extraction']}/validate-extraction",
                json={
                    "entities": entities,
                    "document_type": document_type
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Validation error: {response.text}"
                )
            
            return response.json()
            
    except Exception as e:
        logger.error(f"Error validating extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Enhanced Classification with Entities =====

@app.post("/classify/entities")
async def classify_with_entities(
    file: UploadFile = File(...),
    extract_entities: bool = True,
    quality_check: bool = True,
    apply_preprocessing: bool = True
):
    """Classify document and extract entities with quality check"""
    try:
        content = await file.read()
        result = {
            "filename": file.filename
        }
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Step 1: Quality check if requested
            if quality_check:
                quality_response = await client.post(
                    f"{SERVICES['quality']}/analyze/v2",
                    files={"file": (file.filename, content, file.content_type)},
                    params={"include_preprocessing": apply_preprocessing}
                )
                
                if quality_response.status_code == 200:
                    quality_data = quality_response.json()
                    result["quality_analysis"] = quality_data
                    
                    # If quality is too poor, return early
                    if quality_data.get("verdict") == "poor":
                        result["error"] = "Document quality too poor for processing"
                        return JSONResponse(status_code=400, content=result)
            
            # Step 2: Classification
            class_response = await client.post(
                f"{SERVICES['classification']}/classify",
                files={"file": (file.filename, content, file.content_type)}
            )
            
            if class_response.status_code == 200:
                class_data = class_response.json()
                result["classification"] = class_data
                
                # Step 3: Entity extraction if requested
                if extract_entities:
                    entity_response = await client.post(
                        f"{SERVICES['entity_extraction']}/extract",
                        files={"file": (file.filename, content, file.content_type)},
                        params={
                            "document_type": class_data.get("category"),
                            "extract_tables": True
                        }
                    )
                    
                    if entity_response.status_code == 200:
                        result["entities"] = entity_response.json()
            
            return result
            
    except Exception as e:
        logger.error(f"Error in classification with entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Full Processing Pipeline =====

@app.post("/process/full-pipeline")
async def full_processing_pipeline(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks
):
    """
    Complete document processing pipeline:
    1. Quality analysis
    2. Preprocessing (if needed)
    3. Classification
    4. Entity extraction
    """
    temp_files = []
    
    try:
        content = await file.read()
        result = {
            "filename": file.filename,
            "pipeline_stages": {}
        }
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Stage 1: Quality Analysis
            logger.info("Stage 1: Quality Analysis")
            quality_response = await client.post(
                f"{SERVICES['quality']}/analyze/v2",
                files={"file": (file.filename, content, file.content_type)},
                params={"include_preprocessing": False}
            )
            
            if quality_response.status_code == 200:
                quality_data = quality_response.json()
                result["pipeline_stages"]["quality_analysis"] = quality_data
                
                # Stage 2: Preprocessing if needed
                if quality_data.get("verdict") in ["needs_preprocessing", "poor"]:
                    logger.info("Stage 2: Preprocessing")
                    preprocess_response = await client.post(
                        f"{SERVICES['preprocessing']}/preprocess",
                        files={"file": (file.filename, content, file.content_type)},
                        params={
                            "operations": "all",
                            "return_format": "base64"
                        }
                    )
                    
                    if preprocess_response.status_code == 200:
                        preprocess_data = preprocess_response.json()
                        result["pipeline_stages"]["preprocessing"] = {
                            "applied": True,
                            "operations": preprocess_data.get("operations_applied", [])
                        }
                        # Use preprocessed content for next stages
                        content = base64.b64decode(preprocess_data["data"])
                    else:
                        result["pipeline_stages"]["preprocessing"] = {
                            "applied": False,
                            "error": "Preprocessing failed"
                        }
                else:
                    result["pipeline_stages"]["preprocessing"] = {
                        "applied": False,
                        "reason": "Not needed"
                    }
            
            # Stage 3: Classification
            logger.info("Stage 3: Classification")
            class_response = await client.post(
                f"{SERVICES['classification']}/classify",
                files={"file": (file.filename, content, file.content_type)}
            )
            
            if class_response.status_code == 200:
                class_data = class_response.json()
                result["pipeline_stages"]["classification"] = class_data
                
                # Stage 4: Entity Extraction
                logger.info("Stage 4: Entity Extraction")
                entity_response = await client.post(
                    f"{SERVICES['entity_extraction']}/extract",
                    files={"file": (file.filename, content, file.content_type)},
                    params={
                        "document_type": class_data.get("category"),
                        "extract_tables": True
                    }
                )
                
                if entity_response.status_code == 200:
                    result["pipeline_stages"]["entity_extraction"] = entity_response.json()
                else:
                    result["pipeline_stages"]["entity_extraction"] = {
                        "error": "Entity extraction failed"
                    }
            else:
                result["pipeline_stages"]["classification"] = {
                    "error": "Classification failed"
                }
            
            # Calculate overall success
            stages = result["pipeline_stages"]
            successful_stages = sum(
                1 for stage in stages.values() 
                if isinstance(stage, dict) and "error" not in stage
            )
            result["overall_success"] = successful_stages == len(stages)
            result["success_rate"] = successful_stages / len(stages)
            
            return result
            
    except Exception as e:
        logger.error(f"Error in full pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp files
        background_tasks.add_task(cleanup_temp_files, temp_files)

# ===== Service Discovery Endpoint =====

@app.get("/services")
async def list_services():
    """List all available services and their endpoints"""
    service_info = {}
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        for service_name, service_url in SERVICES.items():
            try:
                # Try to get service info
                response = await client.get(f"{service_url}/")
                if response.status_code == 200:
                    service_info[service_name] = {
                        "url": service_url,
                        "status": "available",
                        "info": response.json()
                    }
                else:
                    service_info[service_name] = {
                        "url": service_url,
                        "status": "error"
                    }
            except:
                service_info[service_name] = {
                    "url": service_url,
                    "status": "unavailable"
                }
    
    return service_info

# ===== Utility Functions =====

def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.error(f"Error cleaning up file {path}: {e}")

# ===== Backwards Compatibility Endpoints =====

@app.post("/process")
async def process_document_legacy(file: UploadFile = File(...)):
    """Legacy endpoint - redirects to full pipeline"""
    return await full_processing_pipeline(file, BackgroundTasks())

@app.post("/classify")
async def classify_document_legacy(file: UploadFile = File(...)):
    """Legacy endpoint - redirects to classification with entities"""
    return await classify_with_entities(file, extract_entities=False)

if __name__ == "__main__":
    logger.info("🚀 Starting Enhanced API Gateway on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
API Gateway for Document Processing Microservices
Orchestrates classification and quality analysis services
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import asyncio
import time
import logging
import uuid
from typing import Optional, Dict, Any
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Processing API Gateway",
    description="Orchestrates document classification and quality analysis services",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service configuration
SERVICES = {
    "quality": {
        "url": "http://localhost:8002", 
        "timeout": 60.0
    },
    "classification": {
        "url": "http://localhost:8001",
        "timeout": 30.0
    }
}

@app.post("/process")
async def process_document_legacy(file: UploadFile = File(...), session_id: Optional[str] = Query(None)):
    """
    Legacy endpoint - redirects to the new versioned endpoint
    Will be deprecated in future versions
    """
    logger.warning("Using deprecated /process endpoint. Please migrate to /api/v1/process-document")
    
    # Forward to the new endpoint with default settings
    return await process_document(
        file=file,
        include_classification=True,
        include_quality=True,
        session_id=session_id
    )

class ServiceClient:
    """Client for communicating with microservices"""
    
    def __init__(self):
        self.client = None
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def call_service(self, service_name: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Call a microservice endpoint"""
        service_config = SERVICES.get(service_name)
        if not service_config:
            raise ValueError(f"Unknown service: {service_name}")
        
        url = f"{service_config['url']}{endpoint}"
        
        try:
            response = await self.client.request(timeout=service_config['timeout'], url=url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail=f"{service_name} service timeout")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"{service_name} service error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"{service_name} service unavailable: {str(e)}")

@app.post("/api/v1/process-document")
async def process_document(
    file: UploadFile = File(...),
    include_classification: bool = Query(True, description="Include document classification"),
    include_quality: bool = Query(True, description="Include quality analysis"),
    session_id: Optional[str] = Query(None, description="Optional session identifier")
):
    """
    Process document with parallel classification and quality analysis
    
    - **file**: Document file to process
    - **include_classification**: Whether to perform classification
    - **include_quality**: Whether to perform quality analysis  
    - **session_id**: Optional session identifier for tracking
    """
    start_time = time.time()
    
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
    
    logger.info(f"Processing document: {file.filename}, Session: {session_id}")
    
    try:
        # Read file content once
        file_content = await file.read()
        
        async with ServiceClient() as client:
            tasks = []
            
            # Prepare file data for requests
            files = {"file": (file.filename, file_content, file.content_type)}
            
            # Parallel service calls
            if include_classification:
                classification_task = asyncio.create_task(
                    client.call_service(
                        "classification",
                        "/classify",
                        method="POST",
                        files=files,
                        data={"session_id": session_id}
                    ),
                    name="classification"
                )
                tasks.append(classification_task)
            
            if include_quality:
                quality_task = asyncio.create_task(
                    client.call_service(
                        "quality",
                        "/analyze",
                        method="POST", 
                        files=files,
                        data={"session_id": session_id}
                    ),
                    name="quality"
                )
                tasks.append(quality_task)
            
            # Wait for all tasks to complete
            results = {}
            errors = {}
            
            if tasks:
                completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(completed_tasks):
                    task_name = tasks[i].get_name()
                    
                    if isinstance(result, Exception):
                        logger.error(f"{task_name} service failed: {result}")
                        errors[task_name] = str(result)
                    else:
                        results[task_name] = result
            
            processing_time = time.time() - start_time
            
            response = {
                "filename": file.filename,
                "file_size_mb": round(len(file_content) / (1024 * 1024), 2),
                "session_id": session_id,
                "processing_time_seconds": round(processing_time, 2),
                "services_requested": [name for name in ["classification", "quality"] 
                                     if (name == "classification" and include_classification) or 
                                        (name == "quality" and include_quality)],
                "services_completed": list(results.keys()),
                "results": results,
                "gateway": "api-gateway",
                "timestamp": time.time()
            }
            
            if errors:
                response["errors"] = errors
                response["partial_success"] = True
            else:
                response["success"] = True
            
            return response
            
    except Exception as e:
        logger.error(f"Gateway error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Gateway error: {str(e)}")

@app.post("/api/v1/classify-only")
async def classify_only(file: UploadFile = File(...), session_id: Optional[str] = None):
    """Classification-only endpoint for faster processing"""
    start_time = time.time()
    
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
    
    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        
        async with ServiceClient() as client:
            result = await client.call_service(
                "classification",
                "/classify",
                method="POST",
                files=files,
                data={"session_id": session_id}
            )
        
        processing_time = time.time() - start_time
        
        return {
            "filename": file.filename,
            "session_id": session_id,
            "processing_time_seconds": round(processing_time, 2),
            "classification": result,
            "service": "classification-only",
            "gateway": "api-gateway"
        }
        
    except Exception as e:
        logger.error(f"Classification-only error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze-quality-only")
async def analyze_quality_only(file: UploadFile = File(...), session_id: Optional[str] = None):
    """Quality analysis-only endpoint"""
    start_time = time.time()
    
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
    
    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        
        async with ServiceClient() as client:
            result = await client.call_service(
                "quality",
                "/analyze",
                method="POST",
                files=files,
                data={"session_id": session_id}
            )
        
        processing_time = time.time() - start_time
        
        return {
            "filename": file.filename,
            "session_id": session_id,
            "processing_time_seconds": round(processing_time, 2),
            "quality_analysis": result,
            "service": "quality-only",
            "gateway": "api-gateway"
        }
        
    except Exception as e:
        logger.error(f"Quality-only error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check health of gateway and all services"""
    gateway_start = time.time()
    
    async with ServiceClient() as client:
        service_health = {}
        
        for service_name, config in SERVICES.items():
            try:
                health_response = await client.call_service(service_name, "/health", method="GET")
                service_health[service_name] = {
                    "status": "healthy",
                    "response": health_response
                }
            except Exception as e:
                service_health[service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        gateway_time = time.time() - gateway_start
        
        # Determine overall status
        all_healthy = all(service["status"] == "healthy" for service in service_health.values())
        
        return {
            "gateway": {
                "status": "healthy",
                "version": "1.0.0",
                "response_time_seconds": round(gateway_time, 3)
            },
            "services": service_health,
            "overall_status": "healthy" if all_healthy else "degraded",
            "timestamp": time.time()
        }

@app.get("/api/v1/service-status")
async def service_status():
    """Detailed service status information"""
    async with ServiceClient() as client:
        status_info = {}
        
        for service_name, config in SERVICES.items():
            try:
                # Try to get service info
                response = await client.call_service(service_name, "/", method="GET")
                status_info[service_name] = {
                    "url": config["url"],
                    "status": "available",
                    "info": response
                }
            except Exception as e:
                status_info[service_name] = {
                    "url": config["url"],
                    "status": "unavailable",
                    "error": str(e)
                }
        
        return {
            "gateway_info": {
                "name": "Document Processing API Gateway",
                "version": "1.0.0",
                "supported_formats": ["PDF", "DOCX", "XLSX", "PNG", "JPG", "TIFF"],
                "endpoints": [
                    "POST /api/v1/process-document - Full document processing",
                    "POST /api/v1/classify-only - Classification only",
                    "POST /api/v1/analyze-quality-only - Quality analysis only",
                    "GET /health - Health check",
                    "GET /api/v1/service-status - Service status"
                ]
            },
            "services": status_info,
            "timestamp": time.time()
        }

@app.get("/")
async def root():
    """Root endpoint with gateway information"""
    return {
        "service": "Document Processing API Gateway",
        "version": "1.0.0",
        "description": "Orchestrates document classification and quality analysis microservices",
        "architecture": "Microservices with MCP integration",
        "endpoints": {
            "POST /api/v1/process-document": "Full document processing (classification + quality)",
            "POST /api/v1/classify-only": "Classification only",
            "POST /api/v1/analyze-quality-only": "Quality analysis only",
            "GET /health": "Health check",
            "GET /api/v1/service-status": "Detailed service status",
            "GET /docs": "Interactive API documentation"
        },
        "performance": {
            "classification_target": "<12 seconds for 32 pages",
            "parallel_processing": True,
            "supported_formats": ["PDF", "DOCX", "XLSX", "PNG", "JPG", "TIFF"]
        },
        "services": {
            "classification": SERVICES["classification"]["url"],
            "quality": SERVICES["quality"]["url"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Document Processing API Gateway...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )

"""
Quality Analysis Microservice with MCP Contracts
Analyzes document quality and provides quality metrics
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import asyncio
import time
import logging
import uuid
from typing import Optional, Dict, Any
from pathlib import Path

# Import our contract definitions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.mcp_contracts import (
    QualityRequest, QualityResponse, QualityResult,
    ServiceHealth, ProcessingStatus
)
from utils.mcp_orchestrator import MCPOrchestrator, ServerConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quality Analysis Microservice with MCP Contracts",
    description="Document quality analysis with formal MCP communication contracts",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global MCP orchestrator
mcp_orchestrator: Optional[MCPOrchestrator] = None

async def get_mcp_orchestrator() -> MCPOrchestrator:
    """Get or create MCP orchestrator instance"""
    global mcp_orchestrator
    if mcp_orchestrator is None:
        mcp_orchestrator = MCPOrchestrator()
        
        # Configure the quality MCP server
        quality_server_config = ServerConfig(
            name="quality_server",
            description="Quality Analysis MCP Server",
            command=["python", "-m", "src.mcp_servers.quality_mcp_server"],
            capabilities=["quality_analysis", "document_scoring"]
        )
        
        await mcp_orchestrator.register_server(quality_server_config)
        
    return mcp_orchestrator

@app.on_event("startup")
async def startup_event():
    """Initialize MCP connections on startup"""
    logger.info("Starting Quality Analysis Microservice...")
    try:
        orchestrator = await get_mcp_orchestrator()
        await orchestrator.start_all_servers()
        logger.info("Quality MCP server connection established")
    except Exception as e:
        logger.error(f"Failed to initialize MCP connections: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up MCP connections on shutdown"""
    logger.info("Shutting down Quality Analysis Microservice...")
    global mcp_orchestrator
    if mcp_orchestrator:
        await mcp_orchestrator.stop_all_servers()
        logger.info("MCP connections closed")

@app.get("/health")
async def health_check() -> ServiceHealth:
    """Health check endpoint"""
    try:
        orchestrator = await get_mcp_orchestrator()
        server_status = await orchestrator.get_server_status("quality_server")
        
        return ServiceHealth(
            service="quality_analysis",
            status="healthy" if server_status else "degraded",
            timestamp=time.time(),
            details={
                "mcp_server_connected": server_status,
                "version": "2.0.0"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return ServiceHealth(
            service="quality_analysis",
            status="unhealthy",
            timestamp=time.time(),
            details={"error": str(e)}
        )

@app.post("/analyze")
async def analyze_document_quality(
    file: UploadFile = File(...),
    analysis_type: str = "comprehensive",
    orchestrator: MCPOrchestrator = Depends(get_mcp_orchestrator)
) -> QualityResponse:
    """
    Analyze document quality using MCP server
    
    Args:
        file: Document file to analyze
        analysis_type: Type of quality analysis to perform
        orchestrator: MCP orchestrator instance
    
    Returns:
        QualityResponse with quality metrics and analysis
    """
    
    request_id = str(uuid.uuid4())
    logger.info(f"Quality analysis request {request_id} started for file: {file.filename}")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Create quality analysis request
        quality_request = QualityRequest(
            request_id=request_id,
            file_path=temp_file_path,
            filename=file.filename,
            analysis_type=analysis_type,
            options={
                "include_readability": True,
                "include_structure": True,
                "include_content_quality": True
            }
        )
        
        # Send request to MCP server
        logger.info(f"Sending quality analysis request to MCP server: {request_id}")
        
        mcp_result = await orchestrator.send_request(
            server_name="quality_server",
            method="analyze_quality",
            params=quality_request.dict()
        )
        
        if not mcp_result.get("success"):
            error_msg = mcp_result.get("error", "Unknown error in quality analysis")
            logger.error(f"Quality analysis failed for {request_id}: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Quality analysis failed: {error_msg}")
        
        # Parse MCP response
        analysis_data = mcp_result.get("result", {})
        
        # Create quality result
        quality_result = QualityResult(
            overall_score=analysis_data.get("overall_score", 0.0),
            readability_score=analysis_data.get("readability_score", 0.0),
            structure_score=analysis_data.get("structure_score", 0.0),
            content_quality_score=analysis_data.get("content_quality_score", 0.0),
            issues=analysis_data.get("issues", []),
            recommendations=analysis_data.get("recommendations", []),
            metrics=analysis_data.get("metrics", {}),
            analysis_timestamp=time.time()
        )
        
        # Create response
        response = QualityResponse(
            request_id=request_id,
            filename=file.filename,
            status=ProcessingStatus.COMPLETED,
            result=quality_result,
            processing_time=analysis_data.get("processing_time", 0.0),
            timestamp=time.time()
        )
        
        logger.info(f"Quality analysis completed successfully for {request_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in quality analysis {request_id}: {e}")
        
        # Return error response
        return QualityResponse(
            request_id=request_id,
            filename=file.filename,
            status=ProcessingStatus.FAILED,
            result=None,
            error=str(e),
            timestamp=time.time()
        )
    finally:
        # Clean up temporary file
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except:
            pass

@app.get("/status/{request_id}")
async def get_analysis_status(
    request_id: str,
    orchestrator: MCPOrchestrator = Depends(get_mcp_orchestrator)
) -> Dict[str, Any]:
    """Get the status of a quality analysis request"""
    
    try:
        # Query MCP server for request status
        mcp_result = await orchestrator.send_request(
            server_name="quality_server",
            method="get_status",
            params={"request_id": request_id}
        )
        
        if mcp_result.get("success"):
            return mcp_result.get("result", {})
        else:
            raise HTTPException(status_code=404, detail="Request not found")
            
    except Exception as e:
        logger.error(f"Error getting status for {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_service_metrics() -> Dict[str, Any]:
    """Get service performance metrics"""
    
    try:
        orchestrator = await get_mcp_orchestrator()
        
        # Get metrics from MCP server
        mcp_result = await orchestrator.send_request(
            server_name="quality_server",
            method="get_metrics",
            params={}
        )
        
        service_metrics = {
            "service": "quality_analysis",
            "version": "2.0.0",
            "uptime": time.time(),
            "mcp_server_metrics": mcp_result.get("result", {}) if mcp_result.get("success") else {}
        }
        
        return service_metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {"error": str(e)}

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Quality Analysis Microservice",
        "version": "2.0.0",
        "description": "Document quality analysis with MCP contracts",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "status": "/status/{request_id}",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Quality Analysis Microservice...")
    uvicorn.run(
        "quality_service:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )

#!/usr/bin/env python3
"""
MCP Orchestrator Microservice
Dedicated service for MCP protocol communication and server management
Port: 8003
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.mcp_orchestrator import MCPOrchestrator, ServerConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="MCP Orchestrator Microservice",
    description="Dedicated service for MCP protocol communication",
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
class MCPCallRequest(BaseModel):
    server_name: str
    tool_name: str
    arguments: Dict[str, Any]

class MCPCallResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    server_name: str
    tool_name: str

class ServerRegistrationRequest(BaseModel):
    name: str
    script_path: str
    timeout: float = 30.0
    max_retries: int = 3

# Global orchestrator
orchestrator: Optional[MCPOrchestrator] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the MCP orchestrator"""
    global orchestrator
    try:
        orchestrator = MCPOrchestrator()
        
        # Register default servers
        servers_dir = project_root / "src" / "mcp_servers"
        
        # Classification MCP server
        orchestrator.register_server(ServerConfig(
            name="classification",
            script_path=str(servers_dir / "classification_mcp_server.py"),
            timeout=30.0,
            max_retries=3
        ))
        
        # Quality MCP server
        orchestrator.register_server(ServerConfig(
            name="quality",
            script_path=str(servers_dir / "quality_mcp_server.py"),
            timeout=30.0,
            max_retries=3
        ))
        
        # Initialize orchestrator
        initialized = await orchestrator.initialize()
        if initialized:
            logger.info("✅ MCP Orchestrator initialized successfully")
        else:
            logger.warning("⚠️ MCP Orchestrator partially initialized")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize MCP orchestrator: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MCP Orchestrator Microservice",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "call": "/call",
            "servers": "/servers",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global orchestrator
    
    server_status = {}
    if orchestrator:
        for server_name in orchestrator.servers:
            server_status[server_name] = {
                "registered": True,
                "initialized": server_name in orchestrator.clients
            }
    
    return {
        "status": "healthy",
        "service": "mcp-orchestrator",
        "timestamp": "2025-09-05T00:00:00Z",
        "servers": server_status,
        "orchestrator_ready": orchestrator is not None
    }

@app.post("/call", response_model=MCPCallResponse)
async def call_mcp_tool(request: MCPCallRequest):
    """
    Call a tool on an MCP server
    """
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="MCP orchestrator not initialized")
    
    try:
        result = await orchestrator.call_tool(
            server_name=request.server_name,
            tool_name=request.tool_name,
            arguments=request.arguments
        )
        
        return MCPCallResponse(
            success=True,
            result=result,
            server_name=request.server_name,
            tool_name=request.tool_name
        )
        
    except Exception as e:
        logger.error(f"❌ MCP call failed: {e}")
        return MCPCallResponse(
            success=False,
            error=str(e),
            server_name=request.server_name,
            tool_name=request.tool_name
        )

@app.post("/servers/register")
async def register_server(request: ServerRegistrationRequest):
    """
    Register a new MCP server
    """
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="MCP orchestrator not initialized")
    
    try:
        config = ServerConfig(
            name=request.name,
            script_path=request.script_path,
            timeout=request.timeout,
            max_retries=request.max_retries
        )
        
        orchestrator.register_server(config)
        
        return {
            "success": True,
            "message": f"Server {request.name} registered successfully",
            "server_name": request.name
        }
        
    except Exception as e:
        logger.error(f"❌ Server registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/servers")
async def list_servers():
    """
    List all registered MCP servers
    """
    global orchestrator
    
    if not orchestrator:
        return {"servers": [], "message": "Orchestrator not initialized"}
    
    servers = []
    for server_name, config in orchestrator.servers.items():
        servers.append({
            "name": server_name,
            "script_path": config.script_path,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
            "initialized": server_name in orchestrator.clients
        })
    
    return {
        "servers": servers,
        "total_count": len(servers)
    }

@app.get("/servers/{server_name}")
async def get_server_info(server_name: str):
    """
    Get information about a specific MCP server
    """
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if server_name not in orchestrator.servers:
        raise HTTPException(status_code=404, detail=f"Server {server_name} not found")
    
    config = orchestrator.servers[server_name]
    
    return {
        "name": server_name,
        "script_path": config.script_path,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
        "initialized": server_name in orchestrator.clients,
        "available_tools": "tools_info_not_implemented"
    }

@app.delete("/servers/{server_name}")
async def unregister_server(server_name: str):
    """
    Unregister an MCP server
    """
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if server_name not in orchestrator.servers:
        raise HTTPException(status_code=404, detail=f"Server {server_name} not found")
    
    try:
        # Close client if exists
        if server_name in orchestrator.clients:
            await orchestrator.clients[server_name].cleanup()
            del orchestrator.clients[server_name]
        
        # Remove from servers
        del orchestrator.servers[server_name]
        
        return {
            "success": True,
            "message": f"Server {server_name} unregistered successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Server unregistration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting MCP Orchestrator Microservice...")
    print("Service: MCP Protocol Communication")
    print("Port: 8003")
    print("Docs: http://localhost:8003/docs")
    print("Health: http://localhost:8003/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        reload=False,
        log_level="info"
    )

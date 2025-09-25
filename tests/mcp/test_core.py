"""
Tests for MCP Core Implementation
"""

import pytest
import json
from datetime import datetime
from src.mcp.core import MCPRequest, MCPResponse, MCPContext, MCPError, MCPServer

@pytest.fixture
def mcp_context():
    return MCPContext.create(metadata={"test": True})

@pytest.fixture
def mcp_request(mcp_context):
    return MCPRequest(
        method="test_method",
        params={"key": "value"},
        context=mcp_context
    )

@pytest.fixture
def mcp_response(mcp_context):
    return MCPResponse(
        result={"status": "success"},
        context=mcp_context
    )

def test_mcp_context_creation():
    """Test MCP context creation"""
    context = MCPContext.create(metadata={"test": True})
    assert isinstance(context.session_id, str)
    assert isinstance(context.timestamp, datetime)
    assert context.metadata == {"test": True}

def test_mcp_request_serialization(mcp_request):
    """Test MCP request serialization/deserialization"""
    # Convert to JSON
    json_str = mcp_request.to_json()
    assert isinstance(json_str, str)
    
    # Parse JSON
    data = json.loads(json_str)
    assert data["jsonrpc"] == "2.0"
    assert data["method"] == "test_method"
    assert data["params"] == {"key": "value"}
    
    # Convert back to request
    new_request = MCPRequest.from_json(json_str)
    assert new_request.method == mcp_request.method
    assert new_request.params == mcp_request.params

def test_mcp_response_serialization(mcp_response):
    """Test MCP response serialization/deserialization"""
    # Convert to JSON
    json_str = mcp_response.to_json()
    assert isinstance(json_str, str)
    
    # Parse JSON
    data = json.loads(json_str)
    assert data["jsonrpc"] == "2.0"
    assert data["result"] == {"status": "success"}
    
    # Convert back to response
    new_response = MCPResponse.from_json(json_str)
    assert new_response.result == mcp_response.result

@pytest.mark.asyncio
async def test_mcp_server():
    """Test MCP server functionality"""
    # Create server
    server = MCPServer("test_server")
    
    # Register method
    async def test_handler(params):
        return {"received": params}
    server.register_method("test_method", test_handler)
    
    # Create request
    request = MCPRequest(method="test_method", params={"test": "data"})
    
    # Handle request
    response = await server.handle_request(request)
    
    assert response.result == {"received": {"test": "data"}}
    assert not response.error

@pytest.mark.asyncio
async def test_mcp_server_method_not_found():
    """Test MCP server method not found error"""
    server = MCPServer("test_server")
    request = MCPRequest(method="nonexistent_method")
    response = await server.handle_request(request)
    
    assert response.error
    assert response.error["code"] == -32601
    assert "not found" in response.error["message"]

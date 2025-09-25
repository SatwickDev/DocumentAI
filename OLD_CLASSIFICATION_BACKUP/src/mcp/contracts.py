"""
Model Context Protocol (MCP) Core Contracts
Base implementation of MCP communication contracts
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import json
from datetime import datetime
import uuid

@dataclass
class MCPMessage:
    """Base message for all MCP communications"""
    jsonrpc: str = "2.0"
    id: Union[int, str] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

@dataclass
class MCPRequest(MCPMessage):
    """MCP request contract"""
    method: str = ""
    params: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps({
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
            "params": self.params or {}
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'MCPRequest':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id", str(uuid.uuid4())),
            method=data.get("method", ""),
            params=data.get("params", {})
        )

@dataclass
class MCPResponse(MCPMessage):
    """MCP response contract"""
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Convert to JSON string"""
        response = {
            "jsonrpc": self.jsonrpc,
            "id": self.id
        }
        if self.error:
            response["error"] = self.error
        else:
            response["result"] = self.result or {}
        return json.dumps(response)

    @classmethod
    def from_json(cls, json_str: str) -> 'MCPResponse':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            result=data.get("result"),
            error=data.get("error")
        )

@dataclass
class MCPTool:
    """MCP tool definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }

@dataclass
class MCPContext:
    """MCP operation context"""
    session_id: str = ""
    timestamp: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "sessionId": self.session_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

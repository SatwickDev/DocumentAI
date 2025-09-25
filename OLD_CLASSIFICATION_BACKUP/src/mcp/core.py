"""
Model Context Protocol (MCP) Core Implementation
"""

from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
import json
from datetime import datetime
import uuid

@dataclass
class MCPContext:
    """MCP Operation Context"""
    session_id: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def create(cls, metadata: Optional[Dict[str, Any]] = None) -> 'MCPContext':
        return cls(
            session_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )

@dataclass
class MCPMessage:
    """Base MCP Message"""
    jsonrpc: str = "2.0"
    id: str = ""
    context: Optional[MCPContext] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.context:
            self.context = MCPContext.create()

@dataclass
class MCPRequest(MCPMessage):
    """MCP Request Message"""
    method: str = ""
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
            "params": self.params or {},
            "context": {
                "session_id": self.context.session_id,
                "timestamp": self.context.timestamp.isoformat(),
                "metadata": self.context.metadata
            } if self.context else {}
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPRequest':
        context_data = data.get("context", {})
        context = MCPContext(
            session_id=context_data.get("session_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(context_data.get("timestamp", datetime.utcnow().isoformat())),
            metadata=context_data.get("metadata", {})
        )
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id", str(uuid.uuid4())),
            method=data.get("method", ""),
            params=data.get("params", {}),
            context=context
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'MCPRequest':
        return cls.from_dict(json.loads(json_str))

@dataclass
class MCPResponse(MCPMessage):
    """MCP Response Message"""
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        response = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "context": {
                "session_id": self.context.session_id,
                "timestamp": self.context.timestamp.isoformat(),
                "metadata": self.context.metadata
            } if self.context else {}
        }
        if self.error:
            response["error"] = self.error
        else:
            response["result"] = self.result or {}
        return response

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPResponse':
        context_data = data.get("context", {})
        context = MCPContext(
            session_id=context_data.get("session_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(context_data.get("timestamp", datetime.utcnow().isoformat())),
            metadata=context_data.get("metadata", {})
        )
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id", str(uuid.uuid4())),
            result=data.get("result"),
            error=data.get("error"),
            context=context
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'MCPResponse':
        return cls.from_dict(json.loads(json_str))

@dataclass
class MCPError:
    """MCP Error Definition"""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        error = {
            "code": self.code,
            "message": self.message
        }
        if self.data:
            error["data"] = self.data
        return error

class MCPServer:
    """Base MCP Server Implementation"""
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.methods: Dict[str, callable] = {}

    def register_method(self, method_name: str, handler: callable):
        """Register a method handler"""
        self.methods[method_name] = handler

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an MCP request"""
        try:
            if request.method not in self.methods:
                return MCPResponse(
                    id=request.id,
                    error=MCPError(
                        code=-32601,
                        message=f"Method '{request.method}' not found"
                    ).to_dict(),
                    context=request.context
                )

            handler = self.methods[request.method]
            result = await handler(request.params or {})
            return MCPResponse(
                id=request.id,
                result=result,
                context=request.context
            )

        except Exception as e:
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=-32000,
                    message=str(e)
                ).to_dict(),
                context=request.context
            )

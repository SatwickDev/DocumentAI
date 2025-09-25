#!/usr/bin/env python3
"""
Simple Classification MCP Server
Provides document classification capabilities via Model Context Protocol
"""

import asyncio
import json
import sys
import os
import tempfile
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

# MCP imports
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    from mcp.server.stdio import stdio_server
except ImportError:
    print("Error: MCP library not installed. Install with: pip install mcp")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleClassificationMCPServer:
    """Simple MCP Server for Document Classification"""
    
    def __init__(self):
        self.server = Server("simple-classification-server")
        self.name = "simple-classification-server"
        self.version = "1.0.0"
        self.setup_tools()
    
    def setup_tools(self):
        """Setup MCP tools for classification"""
        
        @self.server.call_tool()
        async def classify_document(
            file_path: str, 
            config_override: Optional[Dict] = None,
            session_id: Optional[str] = None
        ) -> List[TextContent]:
            """
            Classify a document using simple keyword-based classification
            
            Args:
                file_path: Path to the document file
                config_override: Optional configuration override
                session_id: Session identifier
                
            Returns:
                Classification results as TextContent
            """
            try:
                logger.info(f"Classifying document: {file_path}")
                
                # Simple filename-based classification
                filename = os.path.basename(file_path).lower()
                
                # Basic classification logic
                if any(word in filename for word in ['invoice', 'bill', 'receipt']):
                    category = 'Financial'
                    confidence = 0.85
                    keywords = ['invoice', 'bill', 'receipt']
                elif any(word in filename for word in ['contract', 'agreement', 'legal']):
                    category = 'Legal'
                    confidence = 0.80
                    keywords = ['contract', 'agreement', 'legal']
                elif any(word in filename for word in ['report', 'analysis', 'summary']):
                    category = 'Report'
                    confidence = 0.75
                    keywords = ['report', 'analysis', 'summary']
                else:
                    category = 'Document'
                    confidence = 0.60
                    keywords = ['general']
                
                # Build result
                result = {
                    "category": category,
                    "confidence": confidence,
                    "keywords_found": keywords,
                    "file_path": file_path,
                    "session_id": session_id or "unknown",
                    "processing_time": 0.1,
                    "server": self.name
                }
                
                logger.info(f"Classification result: {category} (confidence: {confidence})")
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
                
            except Exception as e:
                error_result = {
                    "error": str(e),
                    "category": "Error",
                    "confidence": 0.0,
                    "file_path": file_path,
                    "session_id": session_id or "unknown"
                }
                
                logger.error(f"Classification failed: {e}")
                
                return [TextContent(
                    type="text", 
                    text=json.dumps(error_result, indent=2)
                )]
    
    def get_tools(self) -> List[Tool]:
        """Get available tools"""
        return [
            Tool(
                name="classify_document",
                description="Classify a document into categories",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the document file"
                        },
                        "config_override": {
                            "type": "object",
                            "description": "Optional configuration override"
                        },
                        "session_id": {
                            "type": "string", 
                            "description": "Session identifier"
                        }
                    },
                    "required": ["file_path"]
                }
            )
        ]

async def main():
    """Main entry point"""
    logger.info("Starting Simple Classification MCP Server...")
    
    # Create server instance
    server_instance = SimpleClassificationMCPServer()
    
    # Register tools
    @server_instance.server.list_tools()
    async def list_tools() -> List[Tool]:
        return server_instance.get_tools()
    
    logger.info("Simple Classification MCP Server ready for connections")
    
    # Run the server
    async with stdio_server() as streams:
        await server_instance.server.run(
            streams[0], 
            streams[1], 
            server_instance.server.create_initialization_options()
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

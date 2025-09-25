#!/usr/bin/env python3
"""
Document Classification MCP Server
Provides document classification capabilities via Model Context Protocol
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import asyncio
import json
import sys
import os
import tempfile
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

# MCP imports (you'll need to install: pip install mcp)
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent, Resource
    from mcp.server.stdio import stdio_server
except ImportError:
    print("Error: MCP library not installed. Install with: pip install mcp")
    sys.exit(1)

# Import your existing classification logic
try:
    # Import the original classifier (handle filename with spaces)
    import importlib.util
    from pathlib import Path
    classifier_path = Path(__file__).parent.parent / "core" / "document_classifier.py"
    spec = importlib.util.spec_from_file_location("document_classifier", classifier_path)
    dc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc)
    
    # Extract the functions we need
    classify_document_ultra_fast = dc.classify_document_ultra_fast
    load_classification_config = dc.load_classification_config
    PageProcessor = dc.PageProcessor
    
except Exception as e:
    print(f"Error: Could not import document_classifier module: {e}")
    print("Make sure 'document_classifier.py' is in the src/core directory")
    sys.exit(1)

# Setup logging with UTF-8 encoding
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DocumentClassificationMCPServer:
    """MCP Server for Document Classification"""
    
    def __init__(self):
        try:
            self.server = Server("document-classification-server")
            self.name = "document-classification-server"
            self.version = "1.0.0"
            self.config = None
            self.processor = None
            self.initialized = False
            self._init_lock = asyncio.Lock()
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            raise
        
        try:
            # Setup sequence must be in this order
            logger.info("Starting initialization sequence...")

            logger.info("1. Initializing configuration...")
            self.initialize_config()
            logger.info("Configuration initialized")

            logger.info("2. Creating page processor...")
            self.processor = PageProcessor()
            logger.info("Page processor created")

            logger.info("3. Setting up tools...")
            self.setup_tools()
            logger.info("Tools set up")

            logger.info("4. Setting up resources...")
            self.setup_resources()
            logger.info("Resources set up")

            self.initialized = True
            logger.info("[OK] Server initialization complete")

        except Exception as e:
            logger.error(f"[ERROR] Server initialization failed: {e}")
            logger.exception("Detailed error trace:")
            raise
    
    def initialize_config(self):
        """Initialize classification configuration"""
        try:
            self.config = load_classification_config()
            logger.info(f"Loaded {len(self.config.get('categories', {}))} classification categories")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = {"categories": {}, "default_pages_per_pdf": 1}
    
    def setup_tools(self):
        """Setup MCP tools for classification"""
        
        @self.server.call_tool()
        async def classify_document(
            file_path: str, 
            config_override: Optional[Dict] = None,
            session_id: Optional[str] = None
        ) -> List[TextContent]:
            """
            Classify a document using ultra-fast classification
            
            Args:
                file_path: Path to the document file
                config_override: Optional configuration overrides
                session_id: Optional session identifier
            """
            try:
                if not self.initialized:
                    raise Exception("Server not fully initialized")
                
                logger.info(f"Classifying document: {file_path}")
                
                # Validate file exists
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                # Create a mock file object for your existing function
                class MockFile:
                    def __init__(self, path):
                        self.name = path
                
                mock_file = MockFile(file_path)
                
                # Use your existing classification function
                result = classify_document_ultra_fast(mock_file)
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "file_path": file_path,
                        "classification_result": result,
                        "session_id": session_id,
                        "server": "document-classification-mcp",
                        "timestamp": asyncio.get_event_loop().time()
                    }, indent=2)
                )]
                
            except Exception as e:
                logger.error(f"Classification failed for {file_path}: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "file_path": file_path,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "server": "document-classification-mcp"
                    }, indent=2)
                )]
        
        @self.server.call_tool()
        async def classify_text_content(
            text_content: str,
            document_type: str = "unknown",
            page_number: int = 1
        ) -> List[TextContent]:
            """
            Classify text content directly without file processing
            
            Args:
                text_content: Text content to classify
                document_type: Type of document (pdf, docx, etc.)
                page_number: Page number for context
            """
            try:
                # Use the already loaded module
                category = dc.classify_text_ultra_fast(text_content, str(page_number))
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "text_length": len(text_content),
                        "category": category,
                        "document_type": document_type,
                        "page_number": page_number,
                        "server": "document-classification-mcp"
                    }, indent=2)
                )]
                
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": str(e),
                        "server": "document-classification-mcp"
                    }, indent=2)
                )]
        
        @self.server.call_tool()
        async def get_classification_categories() -> List[TextContent]:
            """Get available classification categories and their keywords"""
            try:
                categories_info = {}
                
                for category, keywords in self.config.get("categories", {}).items():
                    categories_info[category] = {
                        "keyword_count": len(keywords),
                        "sample_keywords": [kw.get("keyword", kw) if isinstance(kw, dict) else kw 
                                          for kw in keywords[:5]],
                        "always_separate": self.config.get("always_separate_categories", []),
                        "max_pages": self.config.get("category_page_limits", {}).get(category, 1)
                    }
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "categories": categories_info,
                        "total_categories": len(categories_info),
                        "server": "document-classification-mcp"
                    }, indent=2)
                )]
                
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": str(e),
                        "server": "document-classification-mcp"
                    }, indent=2)
                )]
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available classification tools"""
            return [
                Tool(
                    name="classify_document",
                    description="Classify a document file using ultra-fast OCR and keyword matching. Supports PDF, DOCX, XLSX, and image formats.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string", 
                                "description": "Absolute path to the document file to classify"
                            },
                            "config_override": {
                                "type": "object",
                                "description": "Optional configuration overrides for classification"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Optional session identifier for tracking"
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="classify_text_content", 
                    description="Classify text content directly without file processing",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text_content": {
                                "type": "string",
                                "description": "Text content to classify"
                            },
                            "document_type": {
                                "type": "string",
                                "description": "Type of document (pdf, docx, image, etc.)"
                            },
                            "page_number": {
                                "type": "integer",
                                "description": "Page number for context"
                            }
                        },
                        "required": ["text_content"]
                    }
                ),
                Tool(
                    name="get_classification_categories",
                    description="Get available classification categories and their configuration",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
    
    def setup_resources(self):
        """Setup MCP resources"""
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources"""
            return [
                Resource(
                    uri="config://classification_config",
                    name="Classification Configuration",
                    description="Current classification configuration including categories and keywords",
                    mimeType="application/json"
                ),
                Resource(
                    uri="config://performance_metrics", 
                    name="Performance Metrics",
                    description="Classification performance metrics and targets",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read resource content"""
            if uri == "config://classification_config":
                return json.dumps(self.config, indent=2)
            elif uri == "config://performance_metrics":
                return json.dumps({
                    "target_performance": "<12 seconds for 32 pages",
                    "optimization_features": [
                        "Parallel processing",
                        "Memory-based OCR",
                        "Batch operations",
                        "Smart caching"
                    ],
                    "supported_formats": ["PDF", "DOCX", "XLSX", "PNG", "JPG", "TIFF"]
                }, indent=2)
            else:
                raise ValueError(f"Unknown resource: {uri}")
    
    def list_tools(self):
        """List available tools"""
        return self.server.list_tools()
    
    def get_server(self):
        """Get the underlying MCP server"""
        return self.server

async def main():
    """Main function to run the MCP server"""
    logger.info("Starting Document Classification MCP Server...")
    
    try:
        # Create server instance
        server_instance = DocumentClassificationMCPServer()
        
        # Initialize server
        try:
            # Create server instance
            logger.info("Starting server initialization...")
            
            # Initialize configuration
            server_instance.initialize_config()
            logger.info("Configuration initialized")
            
            # Create page processor
            server_instance.processor = PageProcessor()
            logger.info("Page processor created")
            
            # Setup tools and resources
            server_instance.setup_tools()
            server_instance.setup_resources()
            logger.info("Tools and resources initialized")
            
            # Mark as initialized
            server_instance.initialized = True
            logger.info("Server initialization complete")
            
        except Exception as e:
            logger.error(f"Server initialization failed: {e}")
            logger.exception("Detailed error trace:")
            raise
        
        logger.info("MCP Server ready for connections")
        
        # Run server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP Server ready for connections")
            await server_instance.server.run(
                read_stream,
                write_stream,
                server_instance.server.create_initialization_options()
            )
            
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

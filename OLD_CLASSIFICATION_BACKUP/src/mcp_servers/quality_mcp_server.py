#!/usr/bin/env python3
"""
Quality Analysis MCP Server
Provides document quality analysis capabilities via Model Context Protocol
"""

import asyncio
import json
import sys
import os
import logging
from typing import Any, Dict, List, Optional

# MCP imports
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent, Resource
    from mcp.server.stdio import stdio_server
except ImportError:
    print("Error: MCP library not installed. Install with: pip install mcp")
    sys.exit(1)

# Import your existing quality analysis logic
try:
    # Import the Universal Analyzer (new approach)
    import sys
    import os
    quality_analysis_path = os.path.join(os.path.dirname(__file__), '..', '..', 'quality_analysis_updated')
    sys.path.append(quality_analysis_path)
    
    from universal_analyzer import analyze_pdf_fast_parallel
    from quality_config import load_quality_config, verdict_for_page
    
    print("✅ Universal Analyzer loaded successfully")
    
except Exception as e:
    print(f"Error: Could not import Universal Analyzer: {e}")
    print("Make sure 'quality_analysis_updated' directory exists with universal_analyzer.py")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityAnalysisMCPServer:
    """MCP Server for Document Quality Analysis"""
    
    def __init__(self):
        self.server = Server("quality-analysis-server")
        self.analyzer = None
        self.threshold_manager = None
        self.initialize_components()
        self.setup_tools()
        self.setup_resources()
    
    def initialize_components(self):
        """Initialize quality analysis components"""
        try:
            self.threshold_manager = QualityThresholdManager()
            self.analyzer = UniversalDocumentAnalyzer(
                max_workers=4,
                config_path="TresholdConfig.json"
            )
            logger.info("Quality analysis components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            # Fallback initialization
            self.analyzer = UniversalDocumentAnalyzer()
            self.threshold_manager = QualityThresholdManager()
    
    def setup_tools(self):
        """Setup MCP tools for quality analysis"""
        
        @self.server.call_tool()
        async def analyze_document_quality(
            file_path: str,
            metrics_config: Optional[Dict] = None,
            include_detailed_metrics: bool = True
        ) -> List[TextContent]:
            """
            Analyze document quality with comprehensive metrics
            
            Args:
                file_path: Path to the document file
                metrics_config: Optional configuration for specific metrics
                include_detailed_metrics: Whether to include detailed metric breakdown
            """
            try:
                logger.info(f"Analyzing quality for: {file_path}")
                
                # Validate file exists
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                # Analyze document using your existing logic
                results = self.analyzer.analyze_document(file_path)
                
                if not results:
                    raise ValueError("No analysis results obtained")
                
                # Extract quality information
                analysis_result = results[0] if results else None
                
                response_data = {
                    "success": True,
                    "file_path": file_path,
                    "overall_quality": {
                        "verdict": analysis_result.document_verdict.quality_level if analysis_result and analysis_result.document_verdict else "unknown",
                        "confidence": analysis_result.confidence if analysis_result else 0.0,
                        "action_recommendation": analysis_result.document_verdict.action if analysis_result and analysis_result.document_verdict else "unknown",
                        "message": analysis_result.document_verdict.message if analysis_result and analysis_result.document_verdict else "No message available"
                    },
                    "server": "quality-analysis-mcp",
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                if include_detailed_metrics and analysis_result:
                    response_data["detailed_metrics"] = {
                        "blur_score": analysis_result.metrics.blur_score,
                        "contrast_score": analysis_result.metrics.contrast_score,
                        "noise_level": analysis_result.metrics.noise_level,
                        "sharpness_score": analysis_result.metrics.sharpness_score,
                        "brightness_score": analysis_result.metrics.brightness_score,
                        "skew_angle": analysis_result.metrics.skew_angle,
                        "text_coverage": analysis_result.metrics.text_coverage,
                        "ocr_confidence": analysis_result.metrics.ocr_confidence,
                        "margin_safety": analysis_result.metrics.margin_safety,
                        "duplicate_blank_score": analysis_result.metrics.duplicate_blank_score,
                        "compression_artifact_score": analysis_result.metrics.compression_artifact_score,
                        "page_consistency": analysis_result.metrics.page_consistency,
                        "resolution": analysis_result.metrics.resolution
                    }
                    
                    response_data["issues_detected"] = analysis_result.issues
                    response_data["content_info"] = analysis_result.content_info
                    response_data["processing_time"] = analysis_result.processing_time
                
                return [TextContent(
                    type="text",
                    text=json.dumps(response_data, indent=2)
                )]
                
            except Exception as e:
                logger.error(f"Quality analysis failed for {file_path}: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "file_path": file_path,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "server": "quality-analysis-mcp"
                    }, indent=2)
                )]
        
        @self.server.call_tool()
        async def get_quality_thresholds() -> List[TextContent]:
            """Get current quality assessment thresholds"""
            try:
                config = self.threshold_manager.config
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "quality_metrics": config.get("quality_metrics", {}),
                        "verdict_levels": config.get("overall_document_verdict", {}).get("quality_levels", {}),
                        "critical_metrics": config.get("overall_document_verdict", {}).get("critical_metrics", []),
                        "server": "quality-analysis-mcp"
                    }, indent=2)
                )]
                
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": str(e),
                        "server": "quality-analysis-mcp"
                    }, indent=2)
                )]
        
        @self.server.call_tool()
        async def evaluate_single_metric(
            metric_name: str,
            metric_value: float
        ) -> List[TextContent]:
            """
            Evaluate a single quality metric against thresholds
            
            Args:
                metric_name: Name of the metric to evaluate
                metric_value: Value of the metric
            """
            try:
                evaluation = self.threshold_manager.evaluate_metric(metric_name, metric_value)
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "metric_evaluation": {
                            "metric_name": evaluation.metric_name,
                            "value": evaluation.value,
                            "status": evaluation.status,
                            "normalized_score": evaluation.normalized_score,
                            "threshold": evaluation.threshold,
                            "critical_threshold": evaluation.critical_threshold,
                            "excellent_threshold": evaluation.excellent_threshold,
                            "direction": evaluation.direction
                        },
                        "server": "quality-analysis-mcp"
                    }, indent=2)
                )]
                
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": str(e),
                        "server": "quality-analysis-mcp"
                    }, indent=2)
                )]
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available quality analysis tools"""
            return [
                Tool(
                    name="analyze_document_quality",
                    description="Analyze document quality with 12+ comprehensive metrics including blur, contrast, OCR confidence, and more. Returns quality verdict and recommendations.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Absolute path to the document file to analyze"
                            },
                            "metrics_config": {
                                "type": "object",
                                "description": "Optional configuration for specific metrics"
                            },
                            "include_detailed_metrics": {
                                "type": "boolean",
                                "description": "Whether to include detailed metric breakdown (default: true)"
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="get_quality_thresholds",
                    description="Get current quality assessment thresholds and configuration",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="evaluate_single_metric",
                    description="Evaluate a single quality metric against configured thresholds",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "metric_name": {
                                "type": "string",
                                "description": "Name of the metric to evaluate (e.g., 'blur_score', 'contrast_score')"
                            },
                            "metric_value": {
                                "type": "number",
                                "description": "Value of the metric to evaluate"
                            }
                        },
                        "required": ["metric_name", "metric_value"]
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
                    uri="config://quality_thresholds",
                    name="Quality Thresholds Configuration",
                    description="Current quality assessment thresholds and evaluation criteria",
                    mimeType="application/json"
                ),
                Resource(
                    uri="config://metrics_info",
                    name="Quality Metrics Information", 
                    description="Information about all available quality metrics",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read resource content"""
            if uri == "config://quality_thresholds":
                return json.dumps(self.threshold_manager.config, indent=2)
            elif uri == "config://metrics_info":
                return json.dumps({
                    "available_metrics": [
                        {"name": "blur_score", "description": "Measures image sharpness using Laplacian variance", "unit": "variance_value"},
                        {"name": "contrast_score", "description": "Standard deviation of pixel intensities", "unit": "normalized_value"},
                        {"name": "noise_level", "description": "Estimated noise in the image", "unit": "normalized_value"},
                        {"name": "sharpness_score", "description": "Edge-based sharpness measurement", "unit": "edge_strength"},
                        {"name": "brightness_score", "description": "Average brightness level", "unit": "normalized_value"},
                        {"name": "skew_angle", "description": "Document skew angle", "unit": "degrees"},
                        {"name": "text_coverage", "description": "Percentage of page covered by text", "unit": "percentage"},
                        {"name": "ocr_confidence", "description": "Average OCR confidence score", "unit": "confidence_percentage"},
                        {"name": "margin_safety", "description": "Safety of text margins", "unit": "safety_ratio"},
                        {"name": "duplicate_blank_score", "description": "Non-duplicate, non-blank content score", "unit": "uniqueness_score"},
                        {"name": "compression_artifact_score", "description": "Quality accounting for compression", "unit": "quality_score"},
                        {"name": "page_consistency", "description": "Consistency across document pages", "unit": "consistency_ratio"}
                    ],
                    "quality_levels": ["poor", "acceptable", "good", "excellent"],
                    "supported_formats": ["PDF", "DOCX", "XLSX", "PNG", "JPG", "TIFF"]
                }, indent=2)
            else:
                raise ValueError(f"Unknown resource: {uri}")

async def main():
    """Main function to run the MCP server"""
    logger.info("Starting Quality Analysis MCP Server...")
    
    try:
        # Create server instance
        server_instance = QualityAnalysisMCPServer()
        
        # Run server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Quality Analysis MCP Server ready for connections")
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

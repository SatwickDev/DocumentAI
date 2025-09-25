"""
Enhanced Quality Analysis MCP Server
Provides quality analysis with preprocessing via MCP protocol
"""
import asyncio
import json
import base64
import os
import sys
import tempfile
from typing import Dict, Any, Optional
import logging

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mcp.core import MCPServer, MCPRequest, MCPResponse, MCPError, MCPContext
from core.enhanced_quality_analyzer import EnhancedQualityAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedQualityMCPServer(MCPServer):
    """MCP Server for enhanced quality analysis"""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__("enhanced-quality-analyzer")
        self.analyzer = EnhancedQualityAnalyzer(config_path)
        self.register_handlers()
        
    def register_handlers(self):
        """Register all MCP method handlers"""
        # Register tools
        self.register_method("tools/list", self.handle_tools_list)
        
        # Register quality analysis methods
        self.register_method("analyze_quality_enhanced", self.handle_quality_analysis)
        self.register_method("analyze_preprocessing_needs", self.handle_preprocessing_needs)
        self.register_method("get_quality_config", self.handle_get_config)
        self.register_method("validate_quality_metrics", self.handle_validate_metrics)
        
    async def handle_tools_list(self, request: MCPRequest, context: MCPContext) -> MCPResponse:
        """List available tools"""
        tools = [
            {
                "name": "analyze_quality_enhanced",
                "description": "Analyze document quality with comprehensive metrics and preprocessing options",
                "inputSchema": {
                    "type": "object",
                    "required": ["document"],
                    "properties": {
                        "document": {
                            "type": "string",
                            "format": "base64",
                            "description": "Base64 encoded document"
                        },
                        "apply_preprocessing": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to apply preprocessing if needed"
                        },
                        "save_preprocessed": {
                            "type": "boolean",
                            "default": False,
                            "description": "Whether to save preprocessed document"
                        }
                    }
                }
            },
            {
                "name": "analyze_preprocessing_needs",
                "description": "Analyze document to determine required preprocessing operations",
                "inputSchema": {
                    "type": "object",
                    "required": ["document"],
                    "properties": {
                        "document": {
                            "type": "string",
                            "format": "base64"
                        }
                    }
                }
            }
        ]
        
        return MCPResponse(
            jsonrpc="2.0",
            id=request.id,
            result={"tools": tools}
        )
    
    async def handle_quality_analysis(self, request: MCPRequest, context: MCPContext) -> MCPResponse:
        """Handle quality analysis request"""
        temp_path = None
        
        try:
            params = request.params
            
            # Validate required parameters
            if "document" not in params:
                return MCPResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error=MCPError(
                        code=-32602,
                        message="Invalid params",
                        data="Missing required parameter: document"
                    )
                )
            
            # Decode document
            try:
                document_data = base64.b64decode(params["document"])
            except Exception as e:
                return MCPResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error=MCPError(
                        code=-32602,
                        message="Invalid params",
                        data=f"Failed to decode document: {str(e)}"
                    )
                )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(document_data)
                temp_path = tmp_file.name
            
            # Get parameters
            apply_preprocessing = params.get("apply_preprocessing", True)
            save_preprocessed = params.get("save_preprocessed", False)
            
            # Analyze document
            result = await self.analyzer.analyze_document(
                temp_path,
                apply_preprocessing=apply_preprocessing,
                save_preprocessed=save_preprocessed
            )
            
            # Convert result to dict
            response_data = {
                "total_pages": result.total_pages,
                "overall_score": result.overall_score,
                "verdict": result.verdict,
                "page_analyses": [
                    {
                        "page_number": pa.page_number,
                        "metrics": {
                            "blur_score": pa.metrics.blur_score,
                            "contrast_score": pa.metrics.contrast_score,
                            "noise_level": pa.metrics.noise_level,
                            "sharpness_score": pa.metrics.sharpness_score,
                            "brightness_score": pa.metrics.brightness_score,
                            "skew_angle": pa.metrics.skew_angle,
                            "text_coverage": pa.metrics.text_coverage,
                            "ocr_confidence": pa.metrics.ocr_confidence,
                            "margin_safety": pa.metrics.margin_safety,
                            "edge_crop_score": pa.metrics.edge_crop_score,
                            "shadow_glare_score": pa.metrics.shadow_glare_score,
                            "blank_page_score": pa.metrics.blank_page_score,
                            "resolution_score": pa.metrics.resolution_score
                        },
                        "issues": pa.issues,
                        "recommendations": pa.recommendations,
                        "processing_time": pa.processing_time,
                        "needs_preprocessing": pa.needs_preprocessing,
                        "preprocessing_applied": pa.preprocessing_applied
                    }
                    for pa in result.page_analyses
                ],
                "critical_issues": result.critical_issues,
                "recommendations": result.recommendations,
                "processing_time": result.processing_time,
                "preprocessed_pages": result.preprocessed_pages
            }
            
            return MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                result=response_data
            )
            
        except Exception as e:
            logger.error(f"Error in quality analysis: {e}")
            return MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                error=MCPError(
                    code=-32000,
                    message="Server error",
                    data=str(e)
                )
            )
        
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    
    async def handle_preprocessing_needs(self, request: MCPRequest, context: MCPContext) -> MCPResponse:
        """Analyze preprocessing needs without applying them"""
        temp_path = None
        
        try:
            params = request.params
            
            # Validate and decode document
            if "document" not in params:
                return MCPResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error=MCPError(
                        code=-32602,
                        message="Invalid params",
                        data="Missing required parameter: document"
                    )
                )
            
            document_data = base64.b64decode(params["document"])
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(document_data)
                temp_path = tmp_file.name
            
            # Analyze without preprocessing
            result = await self.analyzer.analyze_document(
                temp_path,
                apply_preprocessing=False,
                save_preprocessed=False
            )
            
            # Determine preprocessing needs
            preprocessing_needed = []
            for page_analysis in result.page_analyses:
                if page_analysis.needs_preprocessing:
                    preprocessing_needed.append({
                        "page_number": page_analysis.page_number,
                        "issues": page_analysis.issues,
                        "recommendations": page_analysis.recommendations
                    })
            
            response_data = {
                "needs_preprocessing": len(preprocessing_needed) > 0,
                "pages_needing_preprocessing": preprocessing_needed,
                "overall_score": result.overall_score,
                "verdict": result.verdict,
                "recommended_operations": self._determine_operations(result)
            }
            
            return MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                result=response_data
            )
            
        except Exception as e:
            logger.error(f"Error analyzing preprocessing needs: {e}")
            return MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                error=MCPError(
                    code=-32000,
                    message="Server error",
                    data=str(e)
                )
            )
        
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    
    async def handle_get_config(self, request: MCPRequest, context: MCPContext) -> MCPResponse:
        """Get current quality configuration"""
        try:
            config = self.analyzer.config
            
            return MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                result={"config": config}
            )
            
        except Exception as e:
            return MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                error=MCPError(
                    code=-32000,
                    message="Server error",
                    data=str(e)
                )
            )
    
    async def handle_validate_metrics(self, request: MCPRequest, context: MCPContext) -> MCPResponse:
        """Validate quality metrics against thresholds"""
        try:
            params = request.params
            
            if "metrics" not in params:
                return MCPResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error=MCPError(
                        code=-32602,
                        message="Invalid params",
                        data="Missing required parameter: metrics"
                    )
                )
            
            metrics = params["metrics"]
            thresholds = self.analyzer.thresholds
            
            # Validate each metric
            validation_results = {}
            for metric_name, value in metrics.items():
                if metric_name in thresholds:
                    threshold = thresholds[metric_name]
                    validation_results[metric_name] = {
                        "value": value,
                        "status": "good" if value >= threshold else "poor",
                        "threshold": threshold
                    }
            
            return MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                result={"validation": validation_results}
            )
            
        except Exception as e:
            return MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                error=MCPError(
                    code=-32000,
                    message="Server error",
                    data=str(e)
                )
            )
    
    def _determine_operations(self, result) -> list:
        """Determine recommended preprocessing operations based on analysis"""
        operations = []
        
        # Check common issues across pages
        blur_issues = sum(1 for pa in result.page_analyses if pa.metrics.blur_score < 0.5)
        contrast_issues = sum(1 for pa in result.page_analyses if pa.metrics.contrast_score < 0.4)
        skew_issues = sum(1 for pa in result.page_analyses if pa.metrics.skew_angle < 0.8)
        noise_issues = sum(1 for pa in result.page_analyses if pa.metrics.noise_level < 0.6)
        
        if blur_issues > len(result.page_analyses) * 0.2:
            operations.append("sharpening")
        if contrast_issues > len(result.page_analyses) * 0.2:
            operations.append("contrast_enhancement")
        if skew_issues > len(result.page_analyses) * 0.2:
            operations.append("deskew")
        if noise_issues > len(result.page_analyses) * 0.2:
            operations.append("denoising")
        
        return operations

async def main():
    """Run the enhanced quality MCP server"""
    server = EnhancedQualityMCPServer()
    logger.info("🚀 Enhanced Quality MCP Server starting...")
    
    # Run server
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
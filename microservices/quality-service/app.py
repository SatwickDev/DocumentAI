#!/usr/bin/env python3
"""
Quality Analysis Microservice
Dedicated service for document quality analysis
Port: 8002
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import uuid
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import uvicorn

# Add project root to path
# In Docker container, project_root should be /app, not going up 3 levels
if "docker" in os.environ.get('DOCKER_ENV', '') or Path(__file__).parent.name == 'app':
    # Running in Docker container
    project_root = Path(__file__).parent  # /app
else:
    # Running locally
    project_root = Path(__file__).parent.parent.parent

sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "quality_analysis_updated"))

# Try to import quality analysis modules (will be handled in startup)
analyze_pdf_fast_parallel = None
load_quality_config = None
get_metric_value = None

# Setup logging with UTF-8 encoding
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_preprocessing_recommendation(verdict, score, metrics):
    """Determine preprocessing recommendations based on quality analysis"""
    verdict_lower = verdict.lower()
    
    if "direct analysis" in verdict_lower and score >= 0.85:
        return {
            "recommended": False,
            "reason": "Excellent quality - no preprocessing needed",
            "operations": [],
            "status": "skip",
            "status_color": "green",
            "status_icon": "✅",
            "estimated_time": 0
        }
    elif "pre-processing" in verdict_lower:
        operations = []
        # Determine specific operations based on metrics 
        if metrics.get("contrast_score", 0) < 0.2:
            operations.append("contrast_enhancement")
        if metrics.get("sharpness_score", 0) < 0.25:
            operations.append("sharpening")
        if metrics.get("brightness_score", 0) < 0.3 or metrics.get("brightness_score", 0) > 0.9:
            operations.append("brightness_adjustment")
        if not operations:  # Default operations
            operations = ["adaptive_enhancement"]
            
        return {
            "recommended": True,
            "reason": "Quality can be improved with preprocessing",
            "operations": operations,
            "status": "enhance",
            "status_color": "yellow",
            "status_icon": "🔧",
            "estimated_time": 15
        }
    elif "azure document analysis" in verdict_lower:
        return {
            "recommended": True,
            "reason": "Medium quality - optimize for OCR processing",
            "operations": ["ocr_optimization", "contrast_enhancement"],
            "status": "ocr_prep",
            "status_color": "orange", 
            "status_icon": "🔄",
            "estimated_time": 20
        }
    elif any(word in verdict_lower for word in ["poor", "rescan", "re-scan"]):
        return {
            "recommended": True,
            "reason": "Poor quality - try enhancement as last resort",
            "operations": ["aggressive_enhancement", "noise_reduction"],
            "status": "last_resort",
            "status_color": "red",
            "status_icon": "⚠️",
            "estimated_time": 25
        }
    elif "reupload" in verdict_lower:
        return {
            "recommended": False,
            "reason": "Quality too poor - recommend reupload",
            "operations": [],
            "status": "reupload",
            "status_color": "red",
            "status_icon": "❌",
            "estimated_time": 0
        }
    else:
        return {
            "recommended": True,
            "reason": "Quality uncertain - apply standard preprocessing",
            "operations": ["adaptive_enhancement"],
            "status": "default",
            "status_color": "blue",
            "status_icon": "🤔",
            "estimated_time": 15
        }


# FastAPI app
app = FastAPI(
    title="Document Quality Analysis Microservice",
    description="Dedicated microservice for document quality analysis",
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "quality-service",
            "version": "1.0.0",
            "analyzer_status": "available" if analyze_pdf_fast_parallel else "not available"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Models
class QualityAnalysisRequest(BaseModel):
    session_id: Optional[str] = None
    analysis_type: str = "full"  # full, quick, detailed

class QualityAnalysisResponse(BaseModel):
    success: bool
    session_id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float

# Global analyzer
analyzer = None

@app.on_event("startup")
async def startup_event():
    """Initialize the service"""
    global analyzer, analyze_pdf_fast_parallel, load_quality_config, get_metric_value
    try:
        # Import universal analyzer components
        from universal_analyzer import analyze_pdf_fast_parallel as analyze_func
        from universal_analyzer import get_metric_value as get_value_func
        from quality_config import verdict_for_page as verdict_func, load_quality_config as load_config_func
        
        # Set global references
        global analyze_pdf_fast_parallel, get_metric_value, load_quality_config, verdict_for_page
        analyze_pdf_fast_parallel = analyze_func
        get_metric_value = get_value_func  
        load_quality_config = load_config_func
        verdict_for_page = verdict_func
        
        # Load quality configuration from the quality_analysis_updated directory
        config_path = project_root / "quality_analysis_updated" / "quality_config.yaml"
        analyzer = load_quality_config(str(config_path))
        logger.info("✅ Universal quality analysis system initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize quality service: {e}")
        analyzer = None
        logger.warning("⚠️ Quality analyzer not available")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Document Quality Analysis Microservice",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.post("/analyze", response_model=QualityAnalysisResponse)
async def analyze_document_quality(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    session_id: Optional[str] = None,
    analysis_type: str = "full"
):
    """
    Analyze document quality
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"🔍 Analyzing quality: {file.filename} (Session: {session_id})")

        # Analyze document quality
        import time
        start_time = time.time()

        try:
            # Use universal analyzer for sophisticated quality analysis
            if analyze_pdf_fast_parallel and analyzer:
                # Analyze document using universal analyzer (single-threaded to avoid Numba conflicts)
                analysis_results = analyze_pdf_fast_parallel(temp_file_path, max_workers=1)

                # Process results
                page_results = []
                total_score = 0.0
                valid_pages = 0

                for page_analysis in analysis_results:
                    if page_analysis and not page_analysis.error and page_analysis.metrics:
                        metrics = page_analysis.metrics

                        # Extract metric values ONLY for graph plotting and display
                        def safe_get_metric_value(metric):
                            if not metric:
                                return 0
                            value = get_metric_value(metric)
                            if isinstance(value, tuple):
                                return max(value) if len(value) >= 2 else (value[0] if len(value) == 1 else 0)
                            return value if isinstance(value, (int, float)) else 0

                        page_metrics = {
                            "blur_score": safe_get_metric_value(metrics.blur_score),
                            "contrast_score": safe_get_metric_value(metrics.contrast_score),
                            "sharpness_score": safe_get_metric_value(metrics.sharpness_score),
                            "brightness_score": safe_get_metric_value(metrics.brightness_score),
                            "edge_crop_score": safe_get_metric_value(metrics.edge_crop_score),
                            "shadow_glare_score": safe_get_metric_value(metrics.shadow_glare_score),
                            "blank_page_score": safe_get_metric_value(metrics.blank_page_score),
                        }

                        # Use Universal Analyzer's verdict calculation
                        metric_dict = {
                            "blur_score": get_metric_value(metrics.blur_score),
                            "contrast_score": get_metric_value(metrics.contrast_score),
                            "noise_level": get_metric_value(metrics.noise_level),
                            "sharpness_score": get_metric_value(metrics.sharpness_score),
                            "brightness_score": get_metric_value(metrics.brightness_score),
                            "skew_angle": get_metric_value(metrics.skew_angle),
                            "edge_crop_score": get_metric_value(metrics.edge_crop_score),
                            "shadow_glare_score": get_metric_value(metrics.shadow_glare_score),
                            "blank_page_score": get_metric_value(metrics.blank_page_score),
                        }
                        
                        # Get Universal Analyzer's verdict and confidence
                        try:
                            page_verdict, per_metric, confidence, confidence_category, recommendations = verdict_for_page(metric_dict, analyzer)
                            
                            # Use Universal Analyzer's confidence as the score (0.0 to 1.0)
                            page_score = confidence if isinstance(confidence, (int, float)) else 0.5
                            
                            logger.info(f"Page {page_analysis.page_num}: Raw Confidence={confidence}, Type={type(confidence)}, Verdict={page_verdict}")
                            logger.info(f"Page {page_analysis.page_num}: Metric values={metric_dict}")
                            logger.info(f"Page {page_analysis.page_num}: Final Score={page_score}")
                        except Exception as e:
                            logger.error(f"Error calculating verdict for page {page_analysis.page_num}: {e}")
                            page_score = 0.5
                            page_verdict = "analysis error"

                        # Determine preprocessing recommendations based on verdict and metrics
                        preprocessing_recommendation = get_preprocessing_recommendation(page_verdict, page_score, page_metrics)

                        page_results.append({
                            "page": page_analysis.page_num,
                            "score": page_score,
                            "metrics": page_metrics,
                            "processing_time": page_analysis.processing_time,
                            "verdict": page_verdict,
                            "preprocessing": preprocessing_recommendation
                        })

                        total_score += page_score
                        valid_pages += 1

                # Use Universal Analyzer's overall assessment
                if valid_pages > 0:
                    overall_score = total_score / valid_pages
                    # Use Universal Analyzer's verdict from the first page (or aggregate if needed)
                    first_page_verdict = page_results[0].get("verdict", "Unknown") if page_results else "Unknown"
                    
                    # Keep Universal Analyzer's native verdicts for accuracy
                    verdict = first_page_verdict
                else:
                    overall_score = 0.0
                    verdict = "Unable to analyze"

                result = {
                    "quality_score": round(overall_score, 3),
                    "verdict": verdict,
                    "analysis_type": analysis_type,
                    "pages_analyzed": valid_pages,
                    "page_results": page_results,
                    "file_size_mb": round(os.path.getsize(temp_file_path) / (1024 * 1024), 2),
                    "processing_time": time.time() - start_time
                }
            else:
                # Fallback to basic analysis if universal analyzer not available
                file_size = os.path.getsize(temp_file_path)
                result = {
                    "quality_score": 0.5,
                    "verdict": "Basic analysis only",
                    "analysis_type": analysis_type,
                    "error": "Universal analyzer not available",
                    "file_size_mb": round(file_size / (1024 * 1024), 2)
                }
        except Exception as e:
            result = {
                "quality_score": 0.0,
                "verdict": "Error",
                "error": str(e),
                "analysis_type": analysis_type
            }

        processing_time = time.time() - start_time

        # Clean up temp file
        if background_tasks:
            background_tasks.add_task(os.unlink, temp_file_path)
        else:
            os.unlink(temp_file_path)

        return QualityAnalysisResponse(
            success=True,
            session_id=session_id,
            result=result,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"❌ Quality analysis failed: {e}")

        # Clean up temp file on error
        try:
            os.unlink(temp_file_path)
        except:
            pass

        return QualityAnalysisResponse(
            success=False,
            session_id=session_id,
            error=str(e),
            processing_time=0.0
        )


@app.get("/config")
async def get_config():
    """Get current quality analysis configuration"""
    return {
        "service": "quality-service",
        "analysis_types": ["full", "quick", "detailed"],
        "supported_formats": ["pdf", "jpg", "png", "tiff"]
    }

@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    return {
        "service": "quality-service",
        "uptime": "running",
        "total_analyses": "metrics_not_implemented",
        "average_processing_time": "metrics_not_implemented"
    }

if __name__ == "__main__":
    print("Starting Quality Analysis Microservice...")
    print("Service: Document Quality Analysis")
    print("Port: 8002")
    print("Docs: http://localhost:8002/docs")
    print("Health: http://localhost:8002/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )

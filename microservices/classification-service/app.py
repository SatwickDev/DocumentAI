#!/usr/bin/env python3
"""
Classification Microservice
Dedicated service for document classification
Port: 8001
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging with UTF-8 encoding first
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

# Add paths to Python path
# In Docker container, the structure is /app/app.py and /app/src/
# When running directly, go up to project root
if "microservices" in str(Path(__file__).parent):
    # Running directly from microservices/classification-service/
    project_root = Path(__file__).parent.parent.parent  # Go up to MCP root
else:
    # Running from Docker container
    project_root = Path(__file__).parent

microservices_path = project_root / "microservices"
src_path = project_root / "src"
# Add the new document_classification_updated path
document_classification_updated_path = project_root / "document_classification_updated"

# Add paths to Python path
for path in [str(project_root), str(src_path), str(microservices_path), str(document_classification_updated_path)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Update PYTHONPATH environment variable
paths = [str(src_path), str(microservices_path)]
pythonpath = os.pathsep.join(paths)
os.environ["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

logger.info(f"Python path: {sys.path}")
logger.info(f"Project root: {project_root}")
logger.info(f"Src path: {src_path}")
logger.info(f"Microservices path: {microservices_path}")

# Now import other dependencies
import asyncio
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import uuid
from typing import Optional, Dict, Any
import uvicorn
import time

# Try importing your hybrid classification modules
try:
    # Import your hybrid classification system
    from documentClassifier import classify_document_optimized, OptimizedMultiTechniqueProcessor
    # Import auto-learning system
    from auto_learning import initialize_auto_learning, add_classification_for_learning, get_auto_learning_stats
    logger.info("✅ Successfully imported hybrid classification modules")
    logger.info("✅ Successfully imported auto-learning system")
except ImportError as e:
    logger.error(f"❌ Failed to import hybrid classification modules: {e}")
    logger.error(f"Python path: {sys.path}")
    raise ImportError("Required hybrid classification modules not found. Ensure document_classification_updated is accessible.")

# Models for API
class ClassificationRequest(BaseModel):
    session_id: Optional[str] = None
    config_override: Optional[Dict[str, Any]] = None

class ClassificationResponse(BaseModel):
    success: bool
    session_id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float

# FastAPI app
app = FastAPI(
    title="Document Classification Microservice",
    description="Dedicated microservice for document classification",
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

# Service state
service_state = {
    "initialized": False,
    "config_loaded": False,
    "processor_ready": False,
    "startup_complete": False
}

# Global configuration
config = None
processor = None
auto_learner = None

async def initialize_ml_models():
    """Initialize ML models - either load existing or train new ones"""
    try:
        model_dir = project_root / "document_classification_updated" / "model"
        classifier_path = model_dir / "classifier.pkl"
        vectorizer_path = model_dir / "vectorizer.pkl"
        
        # Check if models exist
        if classifier_path.exists() and vectorizer_path.exists():
            logger.info("✅ Found existing ML models, loading...")
            # Models will be loaded by your DebugMLModelLoader in the processor
            return
        
        # Models don't exist, need to train
        logger.info("🔄 ML models not found, training new models...")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Import your training modules
        sys.path.append(str(project_root / "document_classification_updated"))
        from training_dataset import datasets
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        import joblib
        
        # Prepare training data
        texts, labels = [], []
        for category, texts_list in datasets.items():
            texts.extend(texts_list)
            labels.extend([category] * len(texts_list))
        
        logger.info(f"Training data: {len(texts)} samples across {len(set(labels))} categories")
        
        # Initialize and train vectorizer
        vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 4),
            min_df=1,
            max_df=0.95,
            lowercase=True,
            stop_words=None,
            sublinear_tf=True,
            strip_accents='unicode'
        )
        
        X = vectorizer.fit_transform(texts)
        logger.info(f"Feature matrix shape: {X.shape}")
        
        # Initialize and train model
        model = LogisticRegression(
            max_iter=5000,
            C=1.0,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs'
        )
        
        model.fit(X, labels)
        
        # Validate model
        cv_scores = cross_val_score(model, X, labels, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f}")
        
        # Save models
        joblib.dump(model, classifier_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        # Save model info
        model_info = {
            "training_samples": len(texts),
            "categories": list(set(labels)),
            "feature_count": X.shape[1],
            "cv_accuracy": cv_scores.mean(),
            "training_accuracy": model.score(X, labels)
        }
        
        with open(model_dir / "model_info.json", "w") as f:
            import json
            json.dump(model_info, f, indent=2, default=str)
        
        logger.info("✅ ML models trained and saved successfully!")
        logger.info(f"Training accuracy: {model.score(X, labels):.3f}")
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f}")
        
    except Exception as e:
        logger.error(f"Failed to initialize ML models: {e}")
        logger.error(f"Error details: {str(e)}")
        # Don't raise - allow service to continue with keyword-only classification


async def classify_document_hybrid(file_path: str, processor, session_id: str, original_filename: str = None) -> Dict[str, Any]:
    """
    Hybrid classification function adapted from your documentClassifier.py
    Uses your OptimizedMultiTechniqueProcessor for enhanced classification
    """
    import fitz
    
    try:
        # Determine file type
        file_ext = file_path.lower().split('.')[-1]
        
        if file_ext == 'pdf':
            # PDF processing using your hybrid approach
            doc = fitz.open(file_path)
            total_pages = len(doc)
            
            results = []
            for page_num in range(min(total_pages, 10)):  # Limit to 10 pages for performance
                page = doc[page_num]
                
                # Use your hybrid processor
                page_result = processor.process_page(page, page_num + 1)
                
                # Handle your 4-technique confidence scoring (e.g., "3/4", "2/4")
                confidence_raw = page_result.confidence_score or "0/4"
                confidence_display = confidence_raw  # Keep original format for display
                
                # Calculate percentage for sorting (e.g., "3/4" = 75%)
                try:
                    if "/" in confidence_raw:
                        votes, total = confidence_raw.split("/")
                        confidence_percent = (float(votes) / float(total)) * 100
                    else:
                        confidence_percent = 0.0
                except:
                    confidence_percent = 0.0
                
                # Get the winning technique from technique results and collect all technique scores
                winning_technique = "hybrid"
                technique_scores = []
                if hasattr(page_result, 'techniques_results') and page_result.techniques_results:
                    # Find the technique with highest confidence
                    best_tech_result = max(page_result.techniques_results, key=lambda x: x.confidence)
                    winning_technique = best_tech_result.technique
                    
                    # Debug: Log OCR text extraction for image-based pages
                    if hasattr(page_result, 'document_type') and page_result.document_type == 'image_based':
                        logger.info(f"🔍 DEBUG Page {page_num + 1} OCR Text Analysis:")
                        for tech_result in page_result.techniques_results:
                            if hasattr(tech_result, 'text_extracted') and tech_result.text_extracted:
                                text_preview = tech_result.text_extracted[:200].replace('\n', ' ')
                                logger.info(f"  📄 {tech_result.technique}: '{text_preview}...'")
                    
                    # Collect all technique scores for detailed analysis
                    technique_scores = [
                        {
                            "technique": tech_result.technique,
                            "category": tech_result.category,
                            "confidence": tech_result.confidence,
                            "processing_time": getattr(tech_result, 'processing_time', 0.0),
                            "error": getattr(tech_result, 'error', None)
                        }
                        for tech_result in page_result.techniques_results
                    ]
                
                results.append({
                    "page_num": page_num + 1,
                    "category": page_result.final_category or "Document",
                    "confidence": confidence_raw,  # Show "3/4" format
                    "confidence_percent": confidence_percent,  # For sorting
                    "technique": winning_technique,
                    "technique_scores": technique_scores,  # Detailed scores from all 4 techniques
                    "details": {
                        "techniques_count": len(getattr(page_result, 'techniques_results', [])),
                        "document_type": getattr(page_result, 'document_type', 'Unknown'),
                        "processing_time": getattr(page_result, 'processing_time', 0.0)
                    }
                })
            
            # Create PDFs with classified document types using your function
            # Use original document name for folder (remove extension)
            if original_filename:
                doc_name = os.path.splitext(original_filename)[0]
            else:
                doc_name = f"document_{session_id}"
            output_dir = f"/app/document_classification_updated/{doc_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert results to PageResult format for your PDF creation function
            page_results = []
            for result_dict in results:
                # Create a simple PageResult-like object
                page_result = type('PageResult', (), {
                    'page_num': result_dict['page_num'],
                    'final_category': result_dict['category'],
                    'confidence_score': result_dict['confidence'],
                    'document_type': result_dict['details']['document_type'],
                    'processing_time': result_dict['details']['processing_time'],
                    'techniques_results': []  # Would need actual ClassificationResult objects
                })()
                page_results.append(page_result)
            
            # Create the classified PDFs using your function
            from documentClassifier import create_pdfs_from_multi_technique_results
            try:
                classification_log = create_pdfs_from_multi_technique_results(
                    doc, page_results, output_dir, session_id
                )
            except FileExistsError as e:
                logger.warning(f"Directory already exists, continuing: {e}")
                classification_log = []  # Empty log, but continue processing
            except Exception as e:
                logger.warning(f"PDF creation failed, continuing with classification: {e}")
                classification_log = []  # Empty log, but continue processing
            
            # Save detailed technique scores to JSON file
            detailed_results = {
                "document_name": doc_name,
                "session_id": session_id,
                "total_pages": len(results),
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "page_results": results,  # Contains all technique_scores for each page
                "summary": {
                    "techniques_used_per_page": [
                        {
                            "page": r["page_num"],
                            "document_type": r["details"]["document_type"],
                            "final_category": r["category"],
                            "final_confidence": r["confidence"],
                            "winning_technique": r["technique"],
                            "technique_count": len(r["technique_scores"])
                        }
                        for r in results
                    ]
                }
            }
            
            # Save to JSON file in the document folder
            json_file_path = os.path.join(output_dir, "classification_details.json")
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved detailed classification results to: {json_file_path}")
            
            doc.close()
            
            # Enhanced return format for multiple PDFs display
            if results:
                # Get overall best result for backward compatibility
                best_result = max(results, key=lambda x: x['confidence_percent'])
                
                # Group results by document type to show PDF summary
                pdf_summary = {}
                for log_entry in classification_log:
                    category = log_entry.get("Final Category", "Unknown")
                    pdf_file = log_entry.get("Output File", "")
                    
                    if category not in pdf_summary:
                        pdf_summary[category] = {
                            "category": category,
                            "pdf_files": [],
                            "page_count": 0,
                            "confidence_scores": [],
                            "avg_confidence": "0/4"
                        }
                    
                    if pdf_file and pdf_file not in [p["filename"] for p in pdf_summary[category]["pdf_files"]]:
                        # Find the full file path
                        pdf_path = os.path.join(output_dir, category.replace(" ", "_"), pdf_file)
                        pdf_summary[category]["pdf_files"].append({
                            "filename": pdf_file,
                            "filepath": pdf_path,
                            "pages": [],
                            "confidence": log_entry.get("Confidence Score", "0/4")
                        })
                    
                    # Add page info
                    page_info = {
                        "page_num": log_entry.get("Page Number", 0),
                        "confidence": log_entry.get("Confidence Score", "0/4"),
                        "processing_time": log_entry.get("Processing Time", "0.0s")
                    }
                    
                    # Find the PDF file to add page to
                    for pdf_file_info in pdf_summary[category]["pdf_files"]:
                        if pdf_file_info["filename"] == pdf_file:
                            pdf_file_info["pages"].append(page_info)
                            break
                    
                    pdf_summary[category]["page_count"] += 1
                    pdf_summary[category]["confidence_scores"].append(log_entry.get("Confidence Score", "0/4"))

                # Calculate average confidence for each category
                for category in pdf_summary:
                    scores = pdf_summary[category]["confidence_scores"]
                    if scores:
                        # Convert "X/4" format to percentage for averaging
                        total_votes = 0
                        total_possible = 0
                        for score in scores:
                            if "/" in str(score):
                                votes, possible = str(score).split("/")
                                total_votes += int(votes)
                                total_possible += int(possible)
                        
                        if total_possible > 0:
                            avg_votes = total_votes / len(scores)
                            pdf_summary[category]["avg_confidence"] = f"{avg_votes:.1f}/4"

                # AUTO-LEARNING: Add successful classifications to learning dataset
                if auto_learner:
                    try:
                        for category_name, category_info in pdf_summary.items():
                            for pdf_file_info in category_info["pdf_files"]:
                                pdf_path = pdf_file_info["filepath"]
                                confidence = pdf_file_info["confidence"]
                                
                                # Add this PDF to learning data
                                add_classification_for_learning(
                                    pdf_path=pdf_path,
                                    category=category_name,
                                    confidence=confidence,
                                    session_id=session_id,
                                    page_results=pdf_file_info.get("pages", [])
                                )
                        
                        logger.info(f"📚 Added {len(pdf_summary)} document(s) to auto-learning system")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to add to auto-learning: {e}")

                return {
                    # MAIN VALUES FOR BACKWARD COMPATIBILITY:
                    "category": best_result['category'],
                    "confidence": best_result['confidence'],
                    "confidence_percentage": f"{best_result['confidence_percent']:.1f}%",
                    "classification": best_result['category'],  # Alias
                    
                    # ENHANCED MULTI-PDF INFORMATION:
                    "multiple_pdfs": len(pdf_summary) > 1,
                    "pdf_summary": pdf_summary,
                    "total_categories": len(pdf_summary),
                    "total_pdfs_created": sum(len(cat_info["pdf_files"]) for cat_info in pdf_summary.values()),
                    
                    # ADDITIONAL INFO:
                    "method": "hybrid_4_technique",
                    "session_id": session_id,
                    "document_name": doc_name,
                    "pages_processed": len(results),
                    "technique_used": best_result['technique'],
                    "detailed_results": results,
                    "hybrid_details": best_result['details'],
                    "output_directory": output_dir,
                    "created_pdfs": classification_log
                }
            else:
                return {
                    "classification": "Unclassified",
                    "confidence": 0.0,
                    "method": "hybrid_multi_technique",
                    "session_id": session_id,
                    "error": "No pages processed"
                }
        
        elif file_ext in ['png', 'jpg', 'jpeg', 'tiff', 'tif']:
            # Image processing
            img_doc = fitz.open()
            img_doc.new_page(width=612, height=792)
            page = img_doc[0]
            
            img_rect = fitz.Rect(0, 0, 612, 792)
            page.insert_image(img_rect, filename=file_path)
            
            # Use your hybrid processor
            page_result = processor.process_page(page, 1)
            
            img_doc.close()
            
            return {
                "classification": page_result.best_category,
                "confidence": page_result.confidence_score,
                "method": "hybrid_multi_technique",
                "session_id": session_id,
                "technique_used": page_result.winning_technique,
                "hybrid_details": {
                    "ml_prediction": getattr(page_result, 'ml_result', {}).get('category', 'N/A'),
                    "ml_confidence": getattr(page_result, 'ml_result', {}).get('confidence', 0.0),
                    "keyword_prediction": getattr(page_result, 'regex_result', {}).get('category', 'N/A'),
                    "keyword_confidence": getattr(page_result, 'regex_result', {}).get('confidence', 0.0)
                }
            }
        
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
            
    except Exception as e:
        logger.error(f"Hybrid classification error: {e}")
        return {
            "classification": "Unclassified",
            "confidence": 0.0,
            "method": "hybrid_multi_technique",
            "session_id": session_id,
            "error": str(e)
        }

@app.get("/auto-learning/stats")
async def auto_learning_stats_endpoint():
    """Get auto-learning system statistics"""
    if auto_learner:
        return get_auto_learning_stats()
    else:
        return {"error": "Auto-learning system not initialized"}


@app.get("/debug/last-classification")
async def get_last_classification_debug():
    """Get detailed technique scores from the last classification"""
    global last_classification_result
    if 'last_classification_result' in globals():
        return last_classification_result
    else:
        return {"error": "No classification results available yet"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not service_state["startup_complete"]:
        raise HTTPException(
            status_code=503,
            detail="Service initialization in progress"
        )
    
    return {
        "status": "healthy",
        "initialized": service_state["initialized"],
        "config_loaded": service_state["config_loaded"],
        "processor_ready": service_state["processor_ready"],
        "startup_complete": service_state["startup_complete"]
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the service with proper error handling"""
    global config, processor, service_state
    
    try:
        logger.info("🚀 Starting Classification Service initialization...")
        
        # Step 1: Load configuration using your hybrid system's loader
        config_path = str(project_root / "document_classification_updated" / "classification_config.json")
        if not (project_root / "document_classification_updated" / "classification_config.json").exists():
            # Fallback to original config  
            config_path = str(project_root / "src" / "core" / "classification_config.json")
            
        from documentClassifier import load_classification_config
        config = load_classification_config(config_path)
        logger.info(f"✅ Configuration loaded from {config_path}")
        service_state["config_loaded"] = True
        
        # Step 2: Initialize ML models first (train/load)
        await initialize_ml_models()
        
        # Step 3: Initialize your hybrid processor
        try:
            # Set the model directory path for ML models
            model_dir = project_root / "document_classification_updated" / "model"
            if model_dir.exists():
                os.environ['MODEL_DIR'] = str(model_dir)
                logger.info(f"✅ ML model directory set: {model_dir}")
            
            # Your config loader already converts keywords properly
            classification_keywords = config.get("categories", {})
            tesseract_configs = config.get("tesseract_configs", [])
            
            processor = OptimizedMultiTechniqueProcessor(classification_keywords, tesseract_configs)
            logger.info("✅ Hybrid OptimizedMultiTechniqueProcessor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize hybrid processor: {e}")
            logger.error(f"Error details: {str(e)}")
            raise Exception(f"Required OptimizedMultiTechniqueProcessor initialization failed: {e}")
            
        if not processor:
            raise Exception("Failed to initialize any processor")
        service_state["processor_ready"] = True
        
        # Step 4: Initialize auto-learning system
        global auto_learner
        try:
            model_dir = str(project_root / "document_classification_updated" / "model")
            learning_data_dir = str(project_root / "document_classification_updated" / "learning_data")
            
            # Ensure directories exist
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(learning_data_dir, exist_ok=True)
            
            auto_learner = initialize_auto_learning(model_dir, learning_data_dir)
            logger.info("✅ Auto-learning system initialized successfully")
            logger.info(f"📊 Learning threshold: 50 samples, Min confidence: 70%")
            logger.info(f"📁 Model dir: {model_dir}")
            logger.info(f"📁 Learning data dir: {learning_data_dir}")
        except Exception as auto_learning_error:
            logger.error(f"⚠️ Auto-learning initialization failed: {auto_learning_error}")
            logger.error("📋 Service will continue without auto-learning")
            auto_learner = None
        
        # Step 5: Final initialization
        service_state["initialized"] = True
        service_state["startup_complete"] = True
        logger.info("✅ Classification Service initialization complete")
        if auto_learner:
            logger.info("🎯 Service ready with hybrid ML+keyword classification + auto-learning")
        else:
            logger.info("🎯 Service ready with hybrid ML+keyword classification (auto-learning disabled)")
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

@app.options("/get-pdf/{session_id}/{filename}")
async def options_pdf_file(session_id: str, filename: str):
    """Handle CORS preflight requests for PDF files"""
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )


@app.get("/get-pdf/{session_id}/{filename}")
async def get_pdf_file(session_id: str, filename: str):
    """Serve classified PDF files for viewing in browser"""
    try:
        # Base output directory for classification results
        base_output_dir = "/app/document_classification_updated"
        
        # Look for the PDF file in various possible locations
        possible_paths = [
            # Direct session-based path
            os.path.join(base_output_dir, f"session_{session_id}", "**", filename),
            # Category-based paths (common categories)
            os.path.join(base_output_dir, "**", filename),
        ]
        
        pdf_path = None
        for pattern in possible_paths:
            import glob
            matches = glob.glob(pattern, recursive=True)
            if matches:
                pdf_path = matches[0]  # Take first match
                break
        
        if not pdf_path or not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=404,
                detail=f"PDF file {filename} not found for session {session_id}"
            )
        
        # Serve the PDF file with streaming for faster loading
        from fastapi.responses import StreamingResponse
        
        def generate_pdf_stream():
            with open(pdf_path, 'rb') as pdf_file:
                while True:
                    chunk = pdf_file.read(8192)  # 8KB chunks for streaming
                    if not chunk:
                        break
                    yield chunk
        
        return StreamingResponse(
            generate_pdf_stream(),
            media_type='application/pdf',
            headers={
                "Content-Disposition": f"inline; filename={filename}",
                "Cache-Control": "no-cache, no-store, must-revalidate",  # Disable caching for security
                "Pragma": "no-cache",
                "Expires": "0",
                "Accept-Ranges": "bytes",  # Enable range requests
                "Access-Control-Allow-Origin": "*",  # Allow cross-origin access
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, Range",
                "X-Content-Type-Options": "nosniff",  # Prevent MIME sniffing
                "X-Frame-Options": "ALLOWALL",  # Allow iframe embedding from anywhere
                "Content-Security-Policy": ""  # Remove restrictive CSP for iframe
            }
        )
        
    except Exception as e:
        logger.error(f"Error serving PDF {filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error serving PDF file: {str(e)}"
        )

@app.post("/classify", response_model=ClassificationResponse)
async def classify_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    session_id: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None
):
    """Classify a document using the current configuration"""
    if not service_state["startup_complete"]:
        raise HTTPException(
            status_code=503,
            detail="Service not fully initialized"
        )
        
    if not processor:
        raise HTTPException(
            status_code=503,
            detail="Document processor not available"
        )

    try:
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Create temporary file with proper extension
        file_extension = os.path.splitext(file.filename)[1] if file.filename else '.pdf'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Process document using your hybrid approach
            start_time = time.time()
            
            try:
                # Use your hybrid classification approach
                result = await classify_document_hybrid(
                    temp_file.name,
                    processor,
                    session_id,
                    original_filename=file.filename
                )
            except Exception as e:
                logger.error(f"Hybrid classification failed: {e}")
                raise e
                    
            processing_time = time.time() - start_time

            # Clean up
            background_tasks.add_task(os.unlink, temp_file.name)

            return ClassificationResponse(
                success=True,
                session_id=session_id,
                result=result,
                processing_time=processing_time
            )

    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@app.get("/serve-pdf/{document_name}/{category}/{filename}")
async def serve_pdf_file(document_name: str, category: str, filename: str):
    """Serve PDF files from document_classification_updated directory structure"""
    try:
        # Construct the file path based on the document classification structure
        base_output_dir = "/app/document_classification_updated"
        pdf_path = os.path.join(base_output_dir, document_name, category, filename)
        
        logger.info(f"Serving PDF file: {pdf_path}")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF file not found: {pdf_path}")
            raise HTTPException(
                status_code=404,
                detail=f"PDF file {filename} not found"
            )
        
        # Check if it's actually a PDF file
        if not pdf_path.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File is not a PDF")
        
        # Serve the PDF file with streaming for faster loading
        from fastapi.responses import StreamingResponse
        
        def generate_pdf_stream():
            with open(pdf_path, 'rb') as pdf_file:
                while True:
                    chunk = pdf_file.read(8192)  # 8KB chunks for streaming
                    if not chunk:
                        break
                    yield chunk
        
        logger.info(f"Successfully serving PDF: {filename}")
        
        return StreamingResponse(
            generate_pdf_stream(),
            media_type='application/pdf',
            headers={
                "Content-Disposition": f"inline; filename={filename}",
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                "Accept-Ranges": "bytes",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, Range",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "ALLOWALL"
            }
        )
        
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        raise HTTPException(status_code=404, detail=f"PDF file not found: {filename}")
    except PermissionError:
        logger.error(f"Permission denied accessing PDF: {pdf_path}")
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        logger.error(f"Error serving PDF {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving PDF: {str(e)}")


if __name__ == "__main__":
    try:
        logger.info("🚀 Starting Classification Service on port 8001...")
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except Exception as e:
        logger.error(f"❌ Failed to start Classification Service: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

#!/usr/bin/env python3
"""
API Gateway Microservice - Enhanced Version
Central gateway that routes requests to appropriate microservices
Port: 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import httpx
import logging
import uvicorn
from typing import Optional, Dict, Any
import os
import json
import json
import time
import asyncio
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="API Gateway - Enhanced",
    description="Central gateway for document processing microservices with enhanced features",
    version="2.0.0",
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

# Service URLs - Updated with new services
SERVICES = {
    "classification": os.getenv("CLASSIFICATION_SERVICE_URL", "http://localhost:8001"),
    "quality": os.getenv("QUALITY_SERVICE_URL", "http://localhost:8002"),
    "preprocessing": os.getenv("PREPROCESSING_SERVICE_URL", "http://localhost:8003"),
    "entity_extraction": os.getenv("ENTITY_SERVICE_URL", "http://localhost:8004"),
    "rule_engine": os.getenv("RULE_ENGINE_SERVICE_URL", "http://localhost:8005"),
    "orchestrator": os.getenv("ORCHESTRATOR_SERVICE_URL", "http://localhost:8006"),
    "notification": os.getenv("NOTIFICATION_SERVICE_URL", "http://localhost:8007")
}

# Service health cache
service_health = {}
last_health_check = 0


def transform_entities_for_frontend(entities, document_type):
    """Transform flat entity dict to nested structure expected by frontend"""
    # Define entity categories based on document type
    entity_categories = {
        "purchase_order": {
            "Basic Information": ["po_number", "po_date", "currency", "total_value"],
            "Parties": ["seller_name", "buyer_name"],
            "Product Details": ["goods_description", "quantity", "unit_price"],
            "Terms": ["delivery_terms", "payment_terms", "governing_law_or_force_majeure"]
        },
        "proforma_invoice": {
            "Invoice Details": ["invoice_number", "invoice_date", "currency", "total_amount"],
            "Parties": ["seller_name", "buyer_name"],
            "Product Details": ["goods_description", "quantity", "unit_price"]
        },
        "default": {
            "Extracted Information": list(entities.keys()) if entities else []
        }
    }
    
    # Get categories for this document type
    categories = entity_categories.get(document_type, entity_categories["default"])
    
    # Transform to frontend format
    formatted_entities = {}
    
    for category, field_names in categories.items():
        category_entities = []
        for field_name in field_names:
            if field_name in entities and entities[field_name]:
                # Convert field name to display label
                label = field_name.replace("_", " ").title()
                entity_data = {
                    "label": label,
                    "value": str(entities[field_name]),
                    "confidence": 0.85  # Default confidence for now
                }
                
                # Check if there's a corresponding bounding box
                bbox_field = f"{field_name}_bbox"
                if bbox_field in entities:
                    entity_data["bbox"] = entities[bbox_field]
                
                category_entities.append(entity_data)
        
        if category_entities:  # Only add non-empty categories
            formatted_entities[category] = category_entities
    
    return formatted_entities

async def check_service_health(service_name: str, url: str) -> bool:
    """Check if a service is healthy"""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{url}/health")
            return response.status_code == 200
    except:
        return False

async def get_services_status():
    """Get current status of all services"""
    global service_health, last_health_check
    current_time = time.time()
    
    # Check every 30 seconds
    if current_time - last_health_check > 30:
        for name, url in SERVICES.items():
            service_health[name] = await check_service_health(name, url)
        last_health_check = current_time
    
    return service_health

def analyze_page_preprocessing(page_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a single page to determine preprocessing requirements
    Based on quality verdict from universal analyzer (4 categories only)
    """
    try:
        verdict = page_result.get('verdict', '').lower()
        score = page_result.get('score', 0.0)
        
        logger.info(f"DEBUG: Analyzing page with verdict='{verdict}', "
                    f"score={score:.1%}")
        
        # Map 4 universal analyzer verdicts to preprocessing decisions
        if verdict == "direct analysis":
            result = {
                "needs_preprocessing": False,
                "reason": f"Excellent quality - ready for direct analysis "
                          f"(Score: {score:.1%})",
                "operations": [],
                "status": "direct_analysis",
                "status_icon": "✅",
                "priority": "none",
                "recommended": False,
                "estimated_time": 0
            }
            logger.info(f"DEBUG: Returning direct_analysis for verdict '{verdict}'")
            return result
        elif verdict == "pre-processing":
            return {
                "needs_preprocessing": True,
                "reason": f"Good quality - preprocessing recommended "
                          f"(Score: {score:.1%})",
                "operations": ["adaptive_preprocessing"],
                "status": "preprocessing",
                "status_icon": "�",
                "priority": "medium",
                "recommended": True,
                "estimated_time": 15
            }
        elif verdict == "azure document analysis":
            return {
                "needs_preprocessing": True,
                "reason": f"Medium quality - Azure OCR recommended "
                          f"(Score: {score:.1%})",
                "operations": ["azure_ocr", "document_analysis"],
                "status": "azure_analysis",
                "status_icon": "🔄",
                "priority": "high",
                "recommended": True,
                "estimated_time": 25
            }
        elif verdict == "reupload":
            return {
                "needs_preprocessing": False,
                "reason": f"Poor quality - document needs to be rescanned "
                          f"(Score: {score:.1%})",
                "operations": [],
                "status": "reupload",
                "status_icon": "❌",
                "priority": "urgent",
                "recommended": False,
                "estimated_time": 0
            }
        else:
            # Unknown verdict - default to skip
            return {
                "needs_preprocessing": False,
                "reason": f"Unknown quality verdict: {verdict} "
                          f"(Score: {score:.1%})",
                "operations": [],
                "status": "unknown",
                "status_icon": "❓",
                "priority": "none",
                "recommended": False,
                "estimated_time": 0
            }
    except Exception as e:
        logger.error(f"Error analyzing page preprocessing: {e}")
        return {
            "needs_preprocessing": False,
            "reason": "Error during analysis - skipping preprocessing",
            "operations": [],
            "status": "error",
            "status_icon": "❓",
            "priority": "none",
            "recommended": False,
            "estimated_time": 0
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "API Gateway - Document Processing System",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "services": "/services",
            "process": "/process",
            "classify": "/classify",
            "analyze-quality": "/analyze-quality",
            "preprocess": "/preprocess",
            "preprocess-selective": "/preprocess/selective",
            "extract-entities": "/extract-entities",
            "extract-entities-batch": "/extract-entities/batch",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    services_status = await get_services_status()
    return {
        "status": "healthy",
        "gateway": "online",
        "services": services_status,
        "timestamp": time.time()
    }

@app.get("/services")
async def get_services():
    """Get list of available services and their status"""
    services_status = await get_services_status()
    return {
        "services": SERVICES,
        "status": services_status,
        "timestamp": time.time()
    }

@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    enable_entity_extraction: bool = Form(True),
    enable_enhanced_quality: bool = Form(True)
):
    """
    Process a document through all available services
    """
    try:
        start_time = time.time()
        logger.info(f"Processing document: {file.filename}")
        
        # Read file content once
        file_content = await file.read()
        
        results = {
            "filename": file.filename,
            "file_size_mb": len(file_content) / (1024 * 1024),
            "session_id": session_id or f"gateway-{int(time.time())}",
            "processing_time_seconds": 0,
            "success": True
        }
        
        # Dictionary to store service results
        service_results = {}
        errors = {}
        
        # Initialize processing history with correct order
        processing_history = []
        
        # Define service calls - start with quality analysis
        service_calls = [
            ("quality", SERVICES["quality"], "/analyze"),
        ]
        
        # Will add preprocessing and classification based on quality results
        
        # Add optional services
        if enable_entity_extraction and service_health.get("entity_extraction", False):
            service_calls.append(("entity_extraction", SERVICES["entity_extraction"], "/extract"))
        
        # Process through services with intelligent conditional logic
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: Always run quality analysis first
            try:
                logger.info("Calling quality service...")
                files = {"file": (file.filename, file_content, file.content_type)}
                data = {"session_id": results["session_id"]}
                
                response = await client.post(
                    f"{SERVICES['quality']}/analyze",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    quality_result = response.json()
                    service_results["quality"] = quality_result
                    logger.info("Quality analysis completed successfully")
                    
                    # Add to processing history
                    processing_history.append({
                        "step": 1,
                        "service": "Quality Analysis",
                        "status": "completed",
                        "timestamp": time.time(),
                        "duration_ms": response.elapsed.total_seconds() * 1000 if response.elapsed else 0
                    })
                    
                    # DEBUG: Log the actual quality result structure
                    logger.info(f"DEBUG: Full quality result: {quality_result}")
                    
                    # Extract quality metrics for decision making
                    # Check if results are nested under a 'result' key
                    if 'result' in quality_result:
                        quality_data = quality_result['result']
                        quality_score = quality_data.get("quality_score", 0.0)
                        verdict = quality_data.get("verdict", "").lower()
                    else:
                        quality_score = quality_result.get("quality_score", 0.0)
                        verdict = quality_result.get("verdict", "").lower()
                    
                    logger.info(f"Quality Score: {quality_score}, Verdict: {verdict}")
                    
                    # Step 2: Analyze per-page preprocessing decisions
                    page_preprocessing_decisions = []
                    if 'result' in quality_result and 'page_results' in quality_result['result']:
                        page_results = quality_result['result']['page_results']
                        for page_result in page_results:
                            page_decision = analyze_page_preprocessing(page_result)
                            page_preprocessing_decisions.append(page_decision)
                    
                    # Step 3: Decide whether to run preprocessing based on quality
                    should_preprocess = False
                    preprocessing_reason = ""
                    
                    if ("excellent" in verdict.lower() or "direct analysis" in verdict.lower()) and quality_score >= 0.85:
                        should_preprocess = False
                        preprocessing_reason = "Skipped - Document quality is excellent (score: {:.1%})".format(quality_score)
                        logger.info(f"🎯 Preprocessing skipped: {preprocessing_reason}")
                    elif "pre-processing" in verdict.lower() or "needs preprocessing" in verdict.lower() or quality_score < 0.85:
                        should_preprocess = True
                        preprocessing_reason = "Applied - Document quality needs improvement (score: {:.1%})".format(quality_score)
                        logger.info(f"🔧 Preprocessing will be applied: {preprocessing_reason}")
                    elif "azure document analysis" in verdict.lower():
                        should_preprocess = True
                        preprocessing_reason = "Applied - Medium quality, suitable for Azure OCR after preprocessing (score: {:.1%})".format(quality_score)
                        logger.info(f"🔧 Preprocessing for Azure OCR: {preprocessing_reason}")
                    elif "poor" in verdict.lower() or "rescan" in verdict.lower() or "re-scan" in verdict.lower():
                        should_preprocess = True  # Try preprocessing as last resort
                        preprocessing_reason = "Applied as last resort - Document quality is poor (score: {:.1%})".format(quality_score)
                        logger.info(f"⚠️ Preprocessing as last resort: {preprocessing_reason}")
                    elif "reupload" in verdict.lower():
                        should_preprocess = False  # Don't waste resources on unusable documents
                        preprocessing_reason = "Skipped - Document quality too poor, recommend reupload (score: {:.1%})".format(quality_score)
                        logger.info(f"❌ Preprocessing skipped: {preprocessing_reason}")
                    else:
                        # Default to preprocessing for unknown cases
                        should_preprocess = True
                        preprocessing_reason = "Applied by default - Unable to determine quality clearly"
                        logger.info(f"🤔 Preprocessing applied by default: {preprocessing_reason}")
                    
                    # Add preprocessing info to results
                    service_results["preprocessing_decision"] = {
                        "applied": should_preprocess,
                        "reason": preprocessing_reason,
                        "based_on_quality_score": quality_score,
                        "based_on_verdict": verdict
                    }
                    
                    # Step 3: Run preprocessing if needed
                    if should_preprocess and service_health.get("preprocessing", False):
                        try:
                            logger.info("Calling preprocessing service...")
                            # Request JSON format to get detailed operation information
                            data_with_format = data.copy() if data else {}
                            data_with_format["return_format"] = "base64"
                            
                            response = await client.post(
                                f"{SERVICES['preprocessing']}/preprocess",
                                files=files,
                                data=data_with_format
                            )
                            
                            if response.status_code == 200:
                                # Handle binary response from preprocessing service
                                content_type = response.headers.get('content-type', '')
                                if 'application/json' in content_type:
                                    service_results["preprocessing"] = response.json()
                                else:
                                    # Binary response (image file)
                                    service_results["preprocessing"] = {
                                        "status": "success",
                                        "content_type": content_type,
                                        "size_bytes": len(response.content),
                                        "format": "binary"
                                    }
                                logger.info("Preprocessing completed successfully")
                                
                                # Add to processing history
                                processing_history.append({
                                    "step": 2,
                                    "service": "Enhanced Preprocessing",
                                    "status": "completed",
                                    "timestamp": time.time(),
                                    "duration_ms": response.elapsed.total_seconds() * 1000 if response.elapsed else 0,
                                    "details": preprocessing_reason
                                })
                                
                            else:
                                try:
                                    error_msg = f"{response.status_code}: {response.text}"
                                except UnicodeDecodeError:
                                    error_msg = f"{response.status_code}: Binary response - cannot decode as text"
                                errors["preprocessing"] = error_msg
                                logger.error(f"Preprocessing failed: {error_msg}")
                                
                                # Add failed preprocessing to history
                                processing_history.append({
                                    "step": 2,
                                    "service": "Enhanced Preprocessing",
                                    "status": "failed",
                                    "timestamp": time.time(),
                                    "error": error_msg
                                })
                                
                        except Exception as e:
                            error_msg = f"Preprocessing service unavailable: {str(e)}"
                            errors["preprocessing"] = error_msg
                            logger.error(error_msg)
                            
                            # Add error to processing history
                            processing_history.append({
                                "step": 2,
                                "service": "Enhanced Preprocessing",
                                "status": "error",
                                "timestamp": time.time(),
                                "error": error_msg
                            })
                            
                    elif should_preprocess:
                        logger.warning("Preprocessing needed but service is unavailable")
                        errors["preprocessing"] = "Service unavailable"
                        
                        # Add unavailable service to history
                        processing_history.append({
                            "step": 2,
                            "service": "Enhanced Preprocessing",
                            "status": "unavailable",
                            "timestamp": time.time(),
                            "error": "Service unavailable"
                        })
                        
                    else:
                        # Add skipped preprocessing to history
                        processing_history.append({
                            "step": 2,
                            "service": "Enhanced Preprocessing",
                            "status": "skipped",
                            "timestamp": time.time(),
                            "details": preprocessing_reason
                        })
                    
                else:
                    error_msg = f"{response.status_code}: {response.text}"
                    errors["quality"] = error_msg
                    logger.error(f"Quality analysis failed: {error_msg}")
                    
            except Exception as e:
                error_msg = f"Quality service unavailable: {str(e)}"
                errors["quality"] = error_msg
                logger.error(error_msg)
            
            # Step 4: Always run classification (regardless of preprocessing)
            try:
                logger.info("Calling classification service...")
                files = {"file": (file.filename, file_content, file.content_type)}
                data = {"session_id": results["session_id"]}
                
                response = await client.post(
                    f"{SERVICES['classification']}/classify",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    service_results["classification"] = response.json()
                    logger.info("Classification completed successfully")
                    
                    # Add to processing history
                    processing_history.append({
                        "step": 3,
                        "service": "Classification",
                        "status": "completed",
                        "timestamp": time.time(),
                        "duration_ms": response.elapsed.total_seconds() * 1000 if response.elapsed else 0
                    })
                    
                else:
                    error_msg = f"{response.status_code}: {response.text}"
                    errors["classification"] = error_msg
                    logger.error(f"Classification failed: {error_msg}")
                    
                    # Add failed classification to history
                    processing_history.append({
                        "step": 3,
                        "service": "Classification",
                        "status": "failed",
                        "timestamp": time.time(),
                        "error": error_msg
                    })
                    
            except Exception as e:
                error_msg = f"Classification service unavailable: {str(e)}"
                errors["classification"] = error_msg
                logger.error(error_msg)
                
                # Add error to processing history
                processing_history.append({
                    "step": 3,
                    "service": "Classification",
                    "status": "error",
                    "timestamp": time.time(),
                    "error": error_msg
                })
        
            # Step 5: Run entity extraction if enabled
            if enable_entity_extraction and service_health.get("entity_extraction", False):
                try:
                    logger.info("Calling entity extraction service...")
                    
                    # Check if we have classified PDFs to process
                    if "classification" in service_results:
                        classification_result = service_results["classification"].get("result", {})
                        pdf_summary = classification_result.get("pdf_summary", {})
                        
                        if pdf_summary:
                            # Use new multi-PDF endpoint with classification results
                            logger.info(f"Processing {len(pdf_summary)} PDF categories from classification")
                            data = {
                                "session_id": results["session_id"],
                                "document_name": classification_result.get("document_name", "unknown"),
                                "pdf_summary": json.dumps(pdf_summary),
                                "output_directory": classification_result.get("output_directory", "")
                            }
                            
                            response = await client.post(
                                f"{SERVICES['entity_extraction']}/extract/multi-pdf",
                                data=data
                            )
                        else:
                            # Fallback to original method if no PDF summary
                            logger.info("No PDF summary found, using original file extraction")
                            files = {"file": (file.filename, file_content, file.content_type)}
                            data = {"session_id": results["session_id"]}
                            
                            # Add document type from classification if available
                            document_type = classification_result.get("classification", "").lower().replace(" ", "_")
                            if document_type:
                                data["document_type"] = document_type
                                logger.info(f"Using classified document type: {document_type}")
                            
                            response = await client.post(
                                f"{SERVICES['entity_extraction']}/extract",
                                files=files,
                                data=data
                            )
                    else:
                        # No classification available, use original method
                        logger.info("No classification available, using original file extraction")
                        files = {"file": (file.filename, file_content, file.content_type)}
                        data = {"session_id": results["session_id"]}
                        
                        response = await client.post(
                            f"{SERVICES['entity_extraction']}/extract",
                            files=files,
                            data=data
                        )
                    
                    if response.status_code == 200:
                        entity_response = response.json()
                        
                        # Handle multi-PDF response format (new enhanced format)
                        if "categories" in entity_response:
                            # This is the new multi-PDF response format
                            service_results["entity_extraction"] = entity_response
                            logger.info(f"Multi-PDF entity extraction: {entity_response.get('total_pdfs_processed', 0)} PDFs processed")
                            
                        # Handle legacy single-file response format  
                        elif "entities" in entity_response:
                            formatted_entities = transform_entities_for_frontend(entity_response["entities"], entity_response.get("document_type", "unknown"))
                            service_results["entity_extraction"] = {
                                **entity_response,
                                "entities": formatted_entities
                            }
                        else:
                            # Fallback - store response as-is
                            service_results["entity_extraction"] = entity_response
                        logger.info("Entity extraction completed successfully")
                        
                        # Add to processing history
                        processing_history.append({
                            "step": 4,
                            "service": "Entities",
                            "status": "completed",
                            "timestamp": time.time(),
                            "duration_ms": response.elapsed.total_seconds() * 1000 if response.elapsed else 0
                        })
                        
                    else:
                        error_msg = f"{response.status_code}: {response.text}"
                        errors["entity_extraction"] = error_msg
                        logger.error(f"Entity extraction failed: {error_msg}")
                        
                        # Add failed entity extraction to history
                        processing_history.append({
                            "step": 4,
                            "service": "Entities",
                            "status": "failed",
                            "timestamp": time.time(),
                            "error": error_msg
                        })
                        
                except Exception as e:
                    error_msg = f"Entity extraction service unavailable: {str(e)}"
                    errors["entity_extraction"] = error_msg
                    logger.error(error_msg)
                    
                    # Add error to processing history
                    processing_history.append({
                        "step": 4,
                        "service": "Entities",
                        "status": "error",
                        "timestamp": time.time(),
                        "error": error_msg
                    })
            else:
                # Add skipped entity extraction to history if not enabled
                if not enable_entity_extraction:
                    processing_history.append({
                        "step": 4,
                        "service": "Entities",
                        "status": "skipped",
                        "timestamp": time.time(),
                        "details": "Entity extraction disabled"
                    })
                else:
                    processing_history.append({
                        "step": 4,
                        "service": "Entities",
                        "status": "unavailable",
                        "timestamp": time.time(),
                        "error": "Service unavailable"
                    })
        
        # Compile results - put service results under 'results' key for frontend
        results["results"] = service_results
        
        # Also maintain top-level results for backward compatibility
        results.update(service_results)
        
        # Flatten entity extraction results to top level for frontend
        if "entity_extraction" in service_results and "entities" in service_results["entity_extraction"]:
            results["entities"] = service_results["entity_extraction"]["entities"]
            
            # Apply rule validation if entities were extracted and we have document classification
            if "entities" in results and "classification" in service_results:
                # Get document type from classification result
                classification_result = service_results["classification"].get("result", {})
                document_type = classification_result.get("classification", "").lower().replace(" ", "_")
                if document_type:
                    try:
                        logger.info(f"Applying rule validation for document type: {document_type}")
                        
                        # Flatten entities for rule validation
                        # Convert from nested format to flat key-value pairs
                        flattened_entities = {}
                        for category, entity_list in results["entities"].items():
                            if isinstance(entity_list, list):
                                for entity in entity_list:
                                    if isinstance(entity, dict) and 'label' in entity and 'value' in entity:
                                        # Convert label to uppercase with underscores
                                        key = entity['label'].upper().replace(' ', '_')
                                        flattened_entities[key] = entity['value']
                        
                        logger.info(f"Flattened entities: {list(flattened_entities.keys())}")
                        
                        # Call rule engine service
                        rule_validation_request = {
                            "entities": flattened_entities,
                            "document_type": document_type,
                            "session_id": results.get("session_id")
                        }
                        
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            rule_response = await client.post(
                                f"{SERVICES['rule_engine']}/validate",
                                json=rule_validation_request
                            )
                            
                            if rule_response.status_code == 200:
                                rule_results = rule_response.json()
                                results["rule_validation"] = rule_results
                                service_results["rule_engine"] = rule_results
                                processing_history.append({
                                    "step": len(processing_history) + 1,
                                    "service": "rule_engine",
                                    "status": "success",
                                    "timestamp": time.time(),
                                    "message": f"Validated {rule_results.get('total_rules', 0)} rules"
                                })
                                logger.info(f"Rule validation completed: {rule_results.get('passed', 0)}/{rule_results.get('total_rules', 0)} rules passed")
                            else:
                                error_msg = f"Rule validation failed with status {rule_response.status_code}"
                                errors["rule_engine"] = error_msg
                                processing_history.append({
                                    "step": len(processing_history) + 1,
                                    "service": "rule_engine",
                                    "status": "error",
                                    "timestamp": time.time(),
                                    "message": error_msg
                                })
                                logger.warning(error_msg)
                                
                    except Exception as e:
                        error_msg = f"Rule validation error: {str(e)}"
                        errors["rule_engine"] = error_msg
                        processing_history.append({
                            "step": len(processing_history) + 1,
                            "service": "rule_engine",
                            "status": "error",
                            "timestamp": time.time(),
                            "message": error_msg
                        })
                        logger.warning(error_msg)
        
        # Add processing history to results
        results["processing_history"] = processing_history
        
        # Add errors if any
        if errors:
            results["errors"] = errors
            results["partial_success"] = True
            
        # Calculate processing time
        results["processing_time_seconds"] = time.time() - start_time
        
        # Return combined results
        return results
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess/selective")
async def selective_preprocess_document(
    file: UploadFile = File(...),
    quality_results: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    """
    Selective preprocessing: Only process pages that need preprocessing
    Creates a mixed PDF with original and processed pages
    """
    try:
        start_time = time.time()
        logger.info(f"Selective preprocessing: {file.filename}")
        
        if not service_health.get("preprocessing", False):
            raise HTTPException(
                status_code=503,
                detail="Preprocessing service is unavailable"
            )
        
        # Validate file format - only PDF supported for selective processing
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension != ".pdf":
            raise HTTPException(
                status_code=415,
                detail="Selective preprocessing only supports PDF files"
            )
        
        # Parse quality results
        try:
            quality_data = json.loads(quality_results)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid quality_results JSON: {e}"
            )
        
        # Check if any pages need preprocessing
        pages_needing_processing = []
        if "page_results" in quality_data:
            for i, page_result in enumerate(quality_data["page_results"]):
                if "preprocessing" in page_result:
                    if page_result["preprocessing"].get("needs_preprocessing", False):
                        pages_needing_processing.append(i + 1)
        
        if not pages_needing_processing:
            # No pages need processing, return original file
            file_content = await file.read()
            return Response(
                content=file_content,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="{file.filename}"'
                }
            )
        
        logger.info(f"Pages needing preprocessing: {pages_needing_processing}")
        
        # Call preprocessing service with selective processing
        file_content = await file.read()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{SERVICES['preprocessing']}/preprocess/selective",
                files={"file": (file.filename, file_content, file.content_type)},
                data={
                    "quality_results": quality_results,
                    "session_id": session_id
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Preprocessing service error: {response.text}"
                )
            
            # Return the processed PDF
            processed_filename = f"{os.path.splitext(file.filename)[0]}_processed.pdf"
            
            # Calculate processing time
            processing_time = time.time() - start_time
            logger.info(f"Selective preprocessing completed in {processing_time:.2f}s")
            
            # Set headers for download
            headers = {
                "Content-Disposition": f'attachment; filename="{processed_filename}"',
                "X-Processing-Time": str(processing_time),
                "X-Pages-Processed": ",".join(map(str, pages_needing_processing))
            }
            
            return Response(
                content=response.content,
                media_type="application/pdf",
                headers=headers
            )
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Preprocessing service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in selective preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify")
async def classify_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Classify a document using the Classification Service"""
    try:
        file_content = await file.read()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SERVICES['classification']}/classify",
                files={"file": (file.filename, file_content, file.content_type)},
                data={"session_id": session_id}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Classification service error: {response.text}"
                )
            
            return response.json()
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Classification service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error classifying document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-quality")
async def analyze_quality(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Analyze document quality"""
    try:
        file_content = await file.read()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SERVICES['quality']}/analyze",
                files={"file": (file.filename, file_content, file.content_type)},
                data={"session_id": session_id}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Quality service error: {response.text}"
                )
            
            quality_results = response.json()
            
            # Add preprocessing analysis for each page
            if "page_results" in quality_results:
                page_count = len(quality_results['page_results'])
                logger.info(f"DEBUG: Adding preprocessing analysis to "
                            f"{page_count} pages")
                for i, page_result in enumerate(quality_results["page_results"]):
                    preprocessing_decision = analyze_page_preprocessing(
                        page_result)
                    page_result["preprocessing"] = preprocessing_decision
                    status = preprocessing_decision.get('status', 'unknown')
                    logger.info(f"DEBUG: Page {i+1} preprocessing decision: "
                                f"{status}")
            
            return quality_results
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Quality service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error analyzing quality: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess")
async def preprocess_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Smart preprocessing-only mode: Run quality analysis first,
    then always apply preprocessing with intelligent reasoning
    """
    try:
        start_time = time.time()
        logger.info(f"Smart preprocessing-only mode: {file.filename}")
        
        if not service_health.get("preprocessing", False):
            raise HTTPException(
                status_code=503,
                detail="Preprocessing service is not available"
            )
            
        file_content = await file.read()
        
        results = {
            "filename": file.filename,
            "file_size_mb": len(file_content) / (1024 * 1024),
            "session_id": session_id or f"preprocess-{int(time.time())}",
            "processing_time_seconds": 0,
            "success": True
        }
        
        # Initialize processing history
        processing_history = []
        service_results = {}
        errors = {}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"file": (file.filename, file_content, file.content_type)}
            data = {"session_id": results["session_id"]}
            
            # Step 1: Run quality analysis first for intelligent decisions
            quality_score = 0.0
            verdict = "unknown"
            
            try:
                logger.info("Running quality analysis first...")
                response = await client.post(
                    f"{SERVICES['quality']}/analyze",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    quality_result = response.json()
                    service_results["quality"] = quality_result
                    logger.info("Quality analysis completed successfully")
                    
                    # Add to processing history
                    processing_history.append({
                        "step": 1,
                        "service": "Quality Analysis",
                        "status": "completed",
                        "timestamp": time.time(),
                        "duration_ms": response.elapsed.total_seconds() * 1000 if response.elapsed else 0
                    })
                    
                    # Extract quality metrics for intelligent reasoning
                    if 'result' in quality_result:
                        quality_data = quality_result['result']
                        quality_score = quality_data.get("quality_score", 0.0)
                        verdict = quality_data.get("verdict", "").lower()
                    else:
                        quality_score = quality_result.get("quality_score", 0.0)
                        verdict = quality_result.get("verdict", "").lower()
                    
                    logger.info(f"Quality Score: {quality_score}, Verdict: {verdict}")
                    
                else:
                    logger.warning("Quality analysis failed, proceeding with preprocessing")
                    processing_history.append({
                        "step": 1,
                        "service": "Quality Analysis",
                        "status": "failed",
                        "timestamp": time.time(),
                        "error": f"{response.status_code}: {response.text}"
                    })
                    
            except Exception as e:
                logger.warning(f"Quality analysis unavailable: {e}, proceeding with preprocessing")
                processing_history.append({
                    "step": 1,
                    "service": "Quality Analysis",
                    "status": "error",
                    "timestamp": time.time(),
                    "error": f"Quality service unavailable: {str(e)}"
                })
            
            # Step 2: Generate intelligent reasoning based on quality
            if "excellent" in verdict and quality_score >= 0.85:
                preprocessing_reason = (f"Applied per user request - Quality is "
                                       f"excellent ({quality_score:.1%}) but "
                                       f"preprocessing was requested")
                logger.info(f"🎯 Smart preprocessing: {preprocessing_reason}")
            elif "needs preprocessing" in verdict or quality_score < 0.85:
                preprocessing_reason = (f"Applied - Document needs improvement "
                                       f"({quality_score:.1%}) and user requested "
                                       f"preprocessing")
                logger.info(f"🔧 Smart preprocessing: {preprocessing_reason}")
            elif "poor" in verdict or "rescan" in verdict:
                preprocessing_reason = (f"Applied - Quality is poor "
                                       f"({quality_score:.1%}) and preprocessing "
                                       f"is recommended")
                logger.info(f"⚠️ Smart preprocessing: {preprocessing_reason}")
            else:
                preprocessing_reason = ("Applied per user request - Quality "
                                       "analysis inconclusive but preprocessing "
                                       "was requested")
                logger.info(f"🤔 Smart preprocessing: {preprocessing_reason}")
            
            # Step 3: Always apply preprocessing (since user requested it) but with intelligent reasoning
            try:
                logger.info("Applying preprocessing with intelligent reasoning...")
                # Request JSON format to get detailed operation information
                data_with_format = data.copy() if data else {}
                data_with_format["return_format"] = "base64"
                
                response = await client.post(
                    f"{SERVICES['preprocessing']}/preprocess",
                    files=files,
                    data=data_with_format
                )
                
                if response.status_code == 200:
                    # Handle binary response from preprocessing service
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' in content_type:
                        preprocessing_result = response.json()
                    else:
                        # Binary response (image file)
                        preprocessing_result = {
                            "status": "success",
                            "content_type": content_type,
                            "size_bytes": len(response.content),
                            "format": "binary"
                        }
                    service_results["preprocessing"] = preprocessing_result
                    logger.info("Preprocessing completed successfully")
                    
                    # Add to processing history
                    processing_history.append({
                        "step": 2,
                        "service": "Enhanced Preprocessing",
                        "status": "completed",
                        "timestamp": time.time(),
                        "duration_ms": response.elapsed.total_seconds() * 1000 if response.elapsed else 0,
                        "details": preprocessing_reason
                    })
                    
                    # Add preprocessing decision info for consistency
                    results["preprocessing_decision"] = {
                        "applied": True,
                        "reason": preprocessing_reason,
                        "based_on_quality_score": quality_score,
                        "based_on_verdict": verdict if verdict != "unknown" else "user_request"
                    }
                    
                else:
                    try:
                        error_msg = f"{response.status_code}: {response.text}"
                    except UnicodeDecodeError:
                        error_msg = f"{response.status_code}: Binary response - cannot decode as text"
                    logger.error(f"Preprocessing failed: {error_msg}")
                    results["success"] = False
                    errors["preprocessing"] = error_msg
                    
                    # Add failed preprocessing to history
                    processing_history.append({
                        "step": 2,
                        "service": "Enhanced Preprocessing",
                        "status": "failed",
                        "timestamp": time.time(),
                        "error": error_msg
                    })
                    
            except Exception as e:
                error_msg = f"Preprocessing service unavailable: {str(e)}"
                logger.error(error_msg)
                results["success"] = False
                errors["preprocessing"] = error_msg
                
                # Add error to processing history
                processing_history.append({
                    "step": 2,
                    "service": "Enhanced Preprocessing",
                    "status": "error",
                    "timestamp": time.time(),
                    "error": error_msg
                })
        
        # Compile results
        results.update(service_results)
        
        # Add processing history to results
        results["processing_history"] = processing_history
        
        # Add errors if any
        if errors:
            results["errors"] = errors
        
        # Calculate processing time
        results["processing_time_seconds"] = time.time() - start_time
        
        return results
            
    except Exception as e:
        logger.error(f"Error in smart preprocessing mode: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-entities")
async def extract_entities(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Extract entities from a document"""
    try:
        if not service_health.get("entity_extraction", False):
            raise HTTPException(
                status_code=503,
                detail="Entity extraction service is not available"
            )
            
        file_content = await file.read()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SERVICES['entity_extraction']}/extract",
                files={"file": (file.filename, file_content, file.content_type)},
                data={"session_id": session_id}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Entity extraction service error: {response.text}"
                )
            
            entity_response = response.json()
            # Transform entity format for frontend
            if "entities" in entity_response:
                formatted_entities = transform_entities_for_frontend(
                    entity_response["entities"], 
                    entity_response.get("document_type", "unknown")
                )
                entity_response["entities"] = formatted_entities
            
            return entity_response
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Entity extraction service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-entities/batch")
async def extract_entities_batch(
    files: list[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    document_type: Optional[str] = Form(None)
):
    """Extract entities from multiple documents with bounding boxes"""
    try:
        if not service_health.get("entity_extraction", False):
            raise HTTPException(
                status_code=503,
                detail="Entity extraction service is not available"
            )
        
        logger.info(f"Processing batch of {len(files)} documents")
        
        # Prepare files for batch processing
        file_data = []
        for file in files:
            content = await file.read()
            file_data.append(("files", (file.filename, content,
                                        file.content_type)))
        
        # Prepare form data
        data = {}
        if session_id:
            data["session_id"] = session_id
        if document_type:
            data["document_type"] = document_type
        
        # Longer timeout for batch processing
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{SERVICES['entity_extraction']}/extract/batch",
                files=file_data,
                data=data
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Batch entity extraction service error: "
                           f"{response.text}"
                )
            
            batch_response = response.json()
            
            # Transform entity format for frontend for each document
            if "results" in batch_response:
                for result in batch_response["results"]:
                    if (result["status"] == "success" and "data" in result
                            and "entities" in result["data"]):
                        formatted_entities = transform_entities_for_frontend(
                            result["data"]["entities"],
                            result["data"].get("document_type", "unknown")
                        )
                        result["data"]["entities"] = formatted_entities
            
            # Add summary statistics
            results = batch_response.get("results", [])
            successful = [r for r in results if r["status"] == "success"]
            failed = [r for r in results if r["status"] == "error"]
            
            batch_response["summary"] = {
                "total_documents": len(files),
                "successful_extractions": len(successful),
                "failed_extractions": len(failed),
                "processing_time": batch_response.get("processing_time", 0),
                "session_id": session_id or f"batch-{int(time.time())}"
            }
            
            return batch_response
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Batch entity extraction service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in batch entity extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate-rules")
async def validate_rules(
    request: Dict[str, Any]
):
    """
    Validate extracted entities against business rules
    Expected format: {
        "entities": {...},
        "document_type": "lc_application",
        "session_id": "optional"
    }
    """
    try:
        # Validate request
        if "entities" not in request or "document_type" not in request:
            raise HTTPException(
                status_code=400,
                detail="Request must include 'entities' and 'document_type'"
            )
        
        # Forward to rule engine service
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SERVICES['rule_engine']}/validate",
                json=request
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Rule engine service error: {response.text}"
                )
            
            return response.json()
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Rule engine service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error validating rules: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported-document-types")
async def get_supported_document_types():
    """Get supported document types for rule validation"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{SERVICES['rule_engine']}/supported-document-types"
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Rule engine service error: {response.text}"
                )
            
            return response.json()
            
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Rule engine service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error getting supported document types: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("API Gateway starting up...")
    logger.info(f"Service URLs: {SERVICES}")
    
    # Initial health check
    await get_services_status()
    logger.info(f"Service health: {service_health}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
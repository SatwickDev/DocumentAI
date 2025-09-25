#!/usr/bin/env python3
"""
Rule Engine Microservice
FastAPI service for document validation using business rules
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from validation_engine import ValidationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Rule Engine Service",
    description="Document validation using business rules",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize validation engine
validation_engine = None

# Pydantic models
class ValidationRequest(BaseModel):
    entities: Dict[str, Any]
    document_type: str
    session_id: Optional[str] = None

class ValidationResponse(BaseModel):
    overall_valid: bool
    document_type: str
    total_rules: int
    passed: int
    failed: int
    validation_time_ms: float
    results: List[Dict[str, Any]]
    session_id: Optional[str] = None

# Document type to rule file mapping
DOCUMENT_RULE_MAPPING = {
    "lc_application": "lc_rules.txt",
    "purchase_order": "purchase_order_rules.txt",
    "proforma_invoice": "proforma_invoice_rules.txt",
    "bank_guarantee": "bank_guarantee_rules.txt",
    "invoice": "invoice_rules.txt"
}

@app.on_event("startup")
async def startup_event():
    """Initialize validation engine on startup"""
    global validation_engine
    try:
        logger.info("🚀 Rule Engine Service starting up...")
        
        # Initialize validation engine
        models_dir = "models"
        cache_dir = "cache"
        validation_engine = ValidationEngine(models_dir=models_dir, cache_dir=cache_dir)
        
        logger.info("✅ Validation engine initialized successfully")
        
        # Check available rule files
        rules_dir = Path("validation_rules")
        if rules_dir.exists():
            rule_files = list(rules_dir.glob("*.txt"))
            logger.info(f"📋 Found {len(rule_files)} rule files: {[f.name for f in rule_files]}")
        else:
            logger.warning("⚠️ Validation rules directory not found")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize validation engine: {e}")
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "rule-engine-service",
        "timestamp": datetime.now().isoformat(),
        "validation_engine_ready": validation_engine is not None
    }

@app.get("/supported-document-types")
async def get_supported_document_types():
    """Get supported document types and their rule files"""
    return {
        "supported_types": list(DOCUMENT_RULE_MAPPING.keys()),
        "rule_mapping": DOCUMENT_RULE_MAPPING
    }

@app.post("/validate")
async def validate_entities(request: ValidationRequest):
    """
    Validate extracted entities against business rules
    """
    try:
        if not validation_engine:
            raise HTTPException(status_code=503, detail="Validation engine not initialized")
        
        # Get document type and corresponding rule file
        document_type = request.document_type.lower()
        if document_type not in DOCUMENT_RULE_MAPPING:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported document type: {document_type}. Supported types: {list(DOCUMENT_RULE_MAPPING.keys())}"
            )
        
        rule_file = DOCUMENT_RULE_MAPPING[document_type]
        rule_path = os.path.join("validation_rules", rule_file)
        
        if not os.path.exists(rule_path):
            raise HTTPException(
                status_code=404,
                detail=f"Rule file not found: {rule_file}. Please ensure rules are defined for {document_type}"
            )
        
        logger.info(f"Validating {document_type} document with {len(request.entities)} entities")
        
        # Convert entity format for validation engine
        # The validation engine expects flat key-value pairs
        validation_data = {}
        
        # Flatten nested entity structure from frontend format
        for category, entity_list in request.entities.items():
            if isinstance(entity_list, list):
                for entity in entity_list:
                    if isinstance(entity, dict) and 'label' in entity and 'value' in entity:
                        # Use label as key, convert to format expected by rules
                        key = entity['label'].upper().replace(' ', '_')
                        validation_data[key] = entity['value']
            else:
                # Handle direct key-value pairs
                validation_data[category] = entity_list
        
        # If entities are already in flat format, use them directly
        if not validation_data:
            validation_data = request.entities
        
        logger.info(f"Prepared validation data with keys: {list(validation_data.keys())}")
        
        # Perform validation
        validation_results = validation_engine.validate_document(
            rule_path, 
            validation_data, 
            document_type.upper()
        )
        
        # Add session_id to results
        validation_results["session_id"] = request.session_id
        
        logger.info(f"Validation completed: {validation_results['passed']}/{validation_results['total_rules']} rules passed")
        
        return validation_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/validate-batch")
async def validate_batch(requests: List[ValidationRequest]):
    """
    Validate multiple documents in batch
    """
    try:
        results = []
        for req in requests:
            try:
                result = await validate_entities(req)
                results.append(result)
            except Exception as e:
                error_result = {
                    "overall_valid": False,
                    "document_type": req.document_type,
                    "error": str(e),
                    "session_id": req.session_id,
                    "total_rules": 0,
                    "passed": 0,
                    "failed": 0,
                    "validation_time_ms": 0,
                    "results": []
                }
                results.append(error_result)
        
        return {
            "batch_results": results,
            "total_documents": len(requests),
            "successful_validations": len([r for r in results if r.get("overall_valid", False)]),
            "failed_validations": len([r for r in results if not r.get("overall_valid", True)])
        }
        
    except Exception as e:
        logger.error(f"Error during batch validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch validation failed: {str(e)}")

@app.get("/rules/{document_type}")
async def get_rules_for_document_type(document_type: str):
    """
    Get all rules for a specific document type
    """
    try:
        document_type = document_type.lower()
        if document_type not in DOCUMENT_RULE_MAPPING:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported document type: {document_type}"
            )
        
        rule_file = DOCUMENT_RULE_MAPPING[document_type]
        rule_path = os.path.join("validation_rules", rule_file)
        
        if not os.path.exists(rule_path):
            raise HTTPException(
                status_code=404,
                detail=f"Rule file not found: {rule_file}"
            )
        
        # Read and parse rules
        rules = []
        with open(rule_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#') and ':' in line:
                    field_tag, rule_text = line.split(':', 1)
                    rules.append({
                        "field": field_tag.strip(),
                        "rule": rule_text.strip(),
                        "line_number": line_num
                    })
        
        return {
            "document_type": document_type,
            "rule_file": rule_file,
            "total_rules": len(rules),
            "rules": rules
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving rules: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve rules: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8005))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
"""
Simple Entity Extraction Microservice
Extracts entities from business documents
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import Optional, Dict, Any, List
import time
import os
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Entity Extraction Service",
    description="Document entity extraction service",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Entity Extraction Service",
        "version": "1.0.0",
        "status": "running",
        "port": 8004
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "entity-extraction",
        "timestamp": time.time(),
        "supported_entities": ["company", "date", "amount", "invoice_number", "po_number"]
    }

@app.post("/extract")
async def extract_entities(
    file: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """Extract entities from a document"""
    try:
        logger.info(f"Extracting entities from: {file.filename}")
        
        # Mock entity extraction results
        entities = {
            "company": [
                {"label": "Vendor", "value": "ABC Corporation", "confidence": 0.95},
                {"label": "Buyer", "value": "XYZ Industries", "confidence": 0.92}
            ],
            "dates": [
                {"label": "Invoice Date", "value": "2024-01-15", "confidence": 0.98},
                {"label": "Due Date", "value": "2024-02-15", "confidence": 0.96}
            ],
            "amounts": [
                {"label": "Total Amount", "value": "$12,345.67", "confidence": 0.99},
                {"label": "Tax Amount", "value": "$1,234.57", "confidence": 0.94}
            ],
            "identifiers": [
                {"label": "Invoice Number", "value": "INV-2024-001", "confidence": 0.97},
                {"label": "PO Number", "value": "PO-2024-12345", "confidence": 0.93}
            ]
        }
        
        return {
            "success": True,
            "filename": file.filename,
            "session_id": session_id or f"extraction-{int(time.time())}",
            "entities": entities,
            "entity_count": sum(len(items) for items in entities.values()),
            "confidence_score": 0.95,
            "processing_time": 2.45
        }
        
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract/batch")
async def extract_entities_batch(files: List[UploadFile] = File(...)):
    """Extract entities from multiple documents"""
    try:
        results = []
        for file in files:
            result = await extract_entities(file)
            results.append(result)
        
        return {
            "success": True,
            "total_files": len(files),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8004))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
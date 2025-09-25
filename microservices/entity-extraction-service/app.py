"""
Entity Extraction Microservice
Extracts structured entities from classified documents
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import Optional, Dict, Any, List
import os
import sys
import tempfile
import logging
from datetime import datetime
import time
import pytesseract
from PIL import Image
import cv2
import numpy as np
import fitz  # PyMuPDF
import re
import json

# Import extractors from the same directory
from purchase_order import PurchaseOrderExtractor
from performa_invoice import PerformaInvoiceExtractor
# from lc_application import LcApplicationExtractor  # Commented out due to PPStructure dependency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Entity Extraction Service",
    description="Extracts structured data from business documents",
    version="1.0.0"
)

# Service configuration
SERVICE_CONFIG = {
    "supported_document_types": [
        "purchase_order",
        "invoice",
        "proforma_invoice",
        "bank_guarantee"
        # "lc_application"  # Temporarily disabled - PPStructure dependency
    ],
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "ocr_languages": ["eng"],
    "confidence_threshold": 0.6
}

# Initialize extractors
EXTRACTORS = {
    "purchase_order": PurchaseOrderExtractor(),
    "proforma_invoice": PerformaInvoiceExtractor(),
    "performa_invoice": PerformaInvoiceExtractor(),  # Alias
    # "lc_application": LcApplicationExtractor(),  # Commented out due to PPStructure dependency
}

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("🚀 Entity Extraction Service starting up...")
    logger.info(f"Supported document types: {SERVICE_CONFIG['supported_document_types']}")
    
    # Check if Tesseract is available
    try:
        pytesseract.get_tesseract_version()
        logger.info("✅ Tesseract OCR is available")
    except Exception as e:
        logger.error(f"❌ Tesseract OCR not found: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "entity-extraction-service",
        "timestamp": datetime.now().isoformat(),
        "extractors_loaded": list(EXTRACTORS.keys())
    }

@app.get("/config")
async def get_config():
    """Get service configuration"""
    return SERVICE_CONFIG

@app.get("/supported-types")
async def get_supported_types():
    """Get list of supported document types with their fields"""
    supported = {}
    
    # Define expected fields for each document type
    supported["purchase_order"] = {
        "fields": [
            "po_number", "po_date", "seller_name", "buyer_name",
            "goods_description", "total_amount", "delivery_terms",
            "payment_terms", "validity_period"
        ],
        "description": "Purchase Order documents"
    }
    
    supported["proforma_invoice"] = {
        "fields": [
            "invoice_number", "invoice_date", "seller_info", "buyer_info",
            "contract_po_number", "incoterm", "destination_port",
            "payment_terms", "country_of_origin", "goods_table"
        ],
        "description": "Proforma Invoice documents"
    }
    
    supported["invoice"] = {
        "fields": [
            "invoice_number", "invoice_date", "seller_info", "buyer_info",
            "line_items", "subtotal", "tax", "total_amount",
            "payment_terms", "due_date"
        ],
        "description": "Standard Invoice documents"
    }
    
    supported["bank_guarantee"] = {
        "fields": [
            "guarantee_number", "issue_date", "beneficiary", "applicant",
            "guarantee_amount", "validity_period", "purpose"
        ],
        "description": "Bank Guarantee documents"
    }
    
    supported["lc_application"] = {
        "fields": [
            "lc_number", "application_date", "applicant", "beneficiary",
            "lc_amount", "expiry_date", "terms_conditions"
        ],
        "description": "Letter of Credit Application"
    }
    
    return supported

@app.post("/extract")
async def extract_entities(
    file: UploadFile = File(...),
    document_type: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    extract_tables: bool = True
):
    """
    Extract entities from a document
    
    Parameters:
    - file: Document file (PDF or image)
    - document_type: Type of document (if known)
    - confidence_threshold: Minimum confidence for extraction
    - extract_tables: Whether to extract tabular data
    """
    temp_path = None
    
    try:
        # Validate file
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        content = await file.read()
        if len(content) > SERVICE_CONFIG["max_file_size"]:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {SERVICE_CONFIG['max_file_size'] / 1024 / 1024}MB"
            )
        
        # Save file temporarily
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        # Extract text for entity recognition
        extracted_text = await extract_text_simple(temp_path)
        
        # Determine document type if not provided
        if not document_type:
            document_type = await classify_document_type(extracted_text)
            logger.info(f"Auto-detected document type: {document_type}")
        
        # Validate document type
        if document_type not in SERVICE_CONFIG["supported_document_types"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported document type: {document_type}"
            )
        
        # Extract entities based on document type
        if document_type in EXTRACTORS:
            # Use specialized extractor
            extractor = EXTRACTORS[document_type]
            entities = extractor.extract(temp_path)
        else:
            # Use generic extraction
            entities = await extract_generic_entities(
                extracted_text, 
                document_type,
                extract_tables
            )
        
        # Calculate confidence score
        confidence = calculate_extraction_confidence(entities)
        
        # Apply confidence threshold
        threshold = confidence_threshold or SERVICE_CONFIG["confidence_threshold"]
        if confidence < threshold:
            logger.warning(f"Low confidence extraction: {confidence} < {threshold}")
        
        # Convert file to base64 image for frontend display
        processed_image_base64, img_width, img_height = convert_file_to_image_base64(temp_path)
        
        return {
            "entities": entities,
            "processed_image": processed_image_base64,
            "image_dimensions": {
                "width": img_width,
                "height": img_height
            }
        }
    
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/extract/multi-pdf")
async def extract_multi_pdf(
    session_id: str = Form(...),
    document_name: str = Form(...),
    pdf_summary: str = Form(...),
    output_directory: str = Form(...)
):
    """
    Extract entities from multiple classified PDFs
    
    Parameters:
    - session_id: Classification session ID
    - document_name: Original document name
    - pdf_summary: JSON string of PDF summary from classification
    - output_directory: Base output directory containing PDFs
    """
    try:
        # Parse the PDF summary
        pdf_data = json.loads(pdf_summary)
        logger.info(f"Processing {len(pdf_data)} PDF categories for document: {document_name}")
        
        results = {
            "session_id": session_id,
            "document_name": document_name,
            "total_categories": len(pdf_data),
            "total_pdfs_processed": 0,
            "categories": {},
            "processing_errors": []
        }
        
        # Process each category of PDFs by reading from the actual folder structure
        base_path = f"/app/document_classification_updated/{document_name}"
        logger.info(f"Looking for PDFs in base path: {base_path}")
        
        if not os.path.exists(base_path):
            error_msg = f"Document folder not found: {base_path}"
            logger.error(error_msg)
            results["processing_errors"].append(error_msg)
            return results
        
        # Get all category folders in the document directory
        category_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        logger.info(f"Found category folders: {category_folders}")
        
        for category_folder in category_folders:
            category_path = os.path.join(base_path, category_folder)
            logger.info(f"Processing category folder: {category_folder}")
            
            category_results = {
                "category": category_folder.replace("_", " "),  # Convert back to readable name
                "pdf_files": [],
                "total_entities": 0,
                "processing_time": 0
            }
            
            # Get all PDF files in this category folder
            try:
                pdf_files = [f for f in os.listdir(category_path) if f.lower().endswith('.pdf')]
                logger.info(f"Found {len(pdf_files)} PDFs in {category_folder}: {pdf_files}")
                
                for pdf_filename in pdf_files:
                    pdf_path = os.path.join(category_path, pdf_filename)
                    logger.info(f"Processing PDF: {pdf_filename} at {pdf_path}")
                    
                    # Check if file exists
                    if not os.path.exists(pdf_path):
                        error_msg = f"PDF file not found: {pdf_path}"
                        logger.error(error_msg)
                        results["processing_errors"].append(error_msg)
                        continue
                
                    try:
                        # Extract text for entity recognition
                        start_time = time.time()
                        extracted_text = await extract_text_simple(pdf_path)
                        
                        # Determine document type from category folder name
                        document_type = category_folder.lower().replace("_", "_")
                        
                        # Extract entities based on document type
                        if document_type in EXTRACTORS:
                            # Use specialized extractor
                            extractor = EXTRACTORS[document_type]
                            entities = extractor.extract(pdf_path)
                        else:
                            # Use generic extraction
                            entities = await extract_generic_entities(
                                extracted_text, 
                                document_type,
                                SERVICE_CONFIG.get("confidence_threshold", 0.6)
                            )
                        
                        # Calculate confidence
                        confidence = calculate_extraction_confidence(entities)
                        processing_time = time.time() - start_time
                        
                        # Convert the processed image to base64
                        processed_image_base64, img_width, img_height = convert_file_to_image_base64(pdf_path)
                        
                        pdf_result = {
                            "filename": pdf_filename,
                            "filepath": pdf_path,
                            "pages": [],
                            "document_type": document_type,
                            "entities": entities,
                            "confidence": confidence,
                            "processing_time": processing_time,
                            "text_length": len(extracted_text),
                            "processed_image": processed_image_base64,
                            "image_dimensions": {
                                "width": img_width,
                                "height": img_height
                            }
                        }
                        
                        category_results["pdf_files"].append(pdf_result)
                        category_results["total_entities"] += len(entities)
                        category_results["processing_time"] += processing_time
                        results["total_pdfs_processed"] += 1
                        
                        logger.info(f"Successfully processed {pdf_filename}: {len(entities)} entities found")
                        
                    except Exception as e:
                        error_msg = f"Error processing {pdf_filename}: {str(e)}"
                        logger.error(error_msg)
                        results["processing_errors"].append(error_msg)
                        
            except Exception as e:
                error_msg = f"Error reading category folder {category_folder}: {str(e)}"
                logger.error(error_msg)
                results["processing_errors"].append(error_msg)
            
            results["categories"][category_folder.replace("_", " ")] = category_results
        
        logger.info(f"Multi-PDF extraction completed: {results['total_pdfs_processed']} PDFs processed")
        return results
        
    except Exception as e:
        logger.error(f"Multi-PDF extraction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Multi-PDF extraction failed: {str(e)}"
        )

@app.post("/extract/batch")
async def extract_batch(
    files: List[UploadFile] = File(...),
    document_type: Optional[str] = None
):
    """Extract entities from multiple documents"""
    results = []
    
    for file in files:
        try:
            result = await extract_entities(
                file=file,
                document_type=document_type
            )
            results.append({
                "filename": file.filename,
                "status": "success",
                "data": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}

@app.post("/validate-extraction")
async def validate_extraction(
    entities: Dict[str, Any],
    document_type: str
):
    """Validate extracted entities against expected schema"""
    try:
        # Get expected fields for document type
        supported_types = await get_supported_types()
        
        if document_type not in supported_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown document type: {document_type}"
            )
        
        expected_fields = supported_types[document_type]["fields"]
        
        # Check which fields are present and valid
        validation_result = {
            "is_valid": True,
            "missing_fields": [],
            "empty_fields": [],
            "field_validation": {}
        }
        
        for field in expected_fields:
            if field not in entities:
                validation_result["missing_fields"].append(field)
                validation_result["is_valid"] = False
            elif not entities[field] or entities[field] == "":
                validation_result["empty_fields"].append(field)
            else:
                # Validate specific field formats
                validation = validate_field(field, entities[field], document_type)
                validation_result["field_validation"][field] = validation
                if not validation["is_valid"]:
                    validation_result["is_valid"] = False
        
        validation_result["completeness_score"] = calculate_completeness_score(
            entities, expected_fields
        )
        
        return validation_result
    
    except Exception as e:
        logger.error(f"Error validating extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

async def extract_text_from_file(file_path: str) -> str:
    """Extract text from PDF or image file using the base extractor logic"""
    # Use the same logic as the base extractor
    try:
        # Import using the simplified path since files are copied to main directory
        from base import BaseExtractor
        base_extractor = BaseExtractor()
        return base_extractor.extract_text(file_path)
    except Exception as e:
        logger.error(f"Error extracting text from file: {e}")
        raise ValueError(f"Failed to extract text from file: {str(e)}")

def convert_file_to_image_base64(file_path: str) -> tuple:
    """Convert PDF or image file to base64 PNG string and return dimensions"""
    try:
        if file_path.lower().endswith('.pdf'):
            # Convert PDF to image using PyMuPDF
            doc = fitz.open(file_path)
            page = doc[0]  # First page
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better quality
            img_data = pix.tobytes("png")
            doc.close()
            
            # Get dimensions
            width = pix.width
            height = pix.height
            
            # Convert to base64
            import base64
            base64_data = base64.b64encode(img_data).decode('utf-8')
            return base64_data, width, height
        else:
            # Handle regular image files
            from PIL import Image
            import io
            import base64
            
            img = Image.open(file_path)
            width, height = img.size
            
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            base64_data = base64.b64encode(img_data).decode('utf-8')
            return base64_data, width, height
    except Exception as e:
        logger.error(f"Error converting file to base64: {str(e)}")
        return "", 0, 0


async def extract_text_simple(file_path: str) -> str:
    """Extract text without coordinates - simplified version"""
    try:
        if file_path.lower().endswith('.pdf'):
            return await extract_pdf_text_simple(file_path)
        else:
            return await extract_image_text_simple(file_path)
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        raise ValueError(f"Failed to extract text: {str(e)}")


async def extract_pdf_text_simple(pdf_path: str) -> str:
    """Extract text from PDF without coordinates"""
    doc = None
    try:
        doc = fitz.open(pdf_path)
        all_text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            all_text += page_text + "\n"
        
        return all_text
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise
    finally:
        if doc:
            doc.close()


async def extract_image_text_simple(image_path: str) -> str:
    """Extract text from image using Tesseract"""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, config='--psm 6')
        return text
    except Exception as e:
        logger.error(f"Error in image OCR: {e}")
        raise


async def extract_pdf_with_coordinates(pdf_path: str) -> Dict[str, Any]:
    """Extract text and coordinates from PDF"""
    doc = None
    try:
        doc = fitz.open(pdf_path)
        all_text = ""
        word_boxes = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get text blocks with coordinates
            blocks = page.get_text("dict")
            page_text = ""
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"]
                            bbox = span["bbox"]  # [x0, y0, x1, y1]
                            
                            # Store word with bounding box (scaled 2x for image)
                            if text.strip():
                                word_boxes.append({
                                    "text": text,
                                    "page": page_num,
                                    "x1": bbox[0] * 2,  # Scale 2x to match image
                                    "y1": bbox[1] * 2,
                                    "x2": bbox[2] * 2,
                                    "y2": bbox[3] * 2,
                                    "confidence": 0.9  # Reliable coordinates
                                })
                                page_text += text
            
            all_text += page_text + "\n"
        
        return {
            "text": all_text,
            "word_boxes": word_boxes,
        }
    except Exception as e:
        logger.error(f"Error processing PDF with coordinates: {e}")
        raise
    finally:
        if doc:
            doc.close()


async def extract_image_with_coordinates(image_path: str) -> Dict[str, Any]:
    """Extract text and coordinates from image using Tesseract"""
    try:
        img = Image.open(image_path)
        
        # Use Tesseract to get text with bounding boxes
        data = pytesseract.image_to_data(img, 
                                        output_type=pytesseract.Output.DICT)
        
        word_boxes = []
        text_parts = []
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:  # Filter out low confidence detections
                text = data['text'][i].strip()
                if text:
                    word_boxes.append({
                        "text": text,
                        "page": 0,  # Single page for images
                        "bbox": {
                            "x": data['left'][i],
                            "y": data['top'][i],
                            "width": data['width'][i],
                            "height": data['height'][i]
                        },
                        "confidence": data['conf'][i] / 100.0
                    })
                    text_parts.append(text)
        
        return {
            "text": " ".join(text_parts),
            "word_boxes": word_boxes,
            "page_count": 1
        }
        
    except Exception as e:
        logger.error(f"Error in image OCR with coordinates: {e}")
        raise


def add_bounding_boxes_to_entities(entities: Dict[str, Any], 
                                  word_boxes: List[Dict]) -> Dict[str, Any]:
    """Add bounding box coordinates to extracted entities"""
    if not word_boxes:
        return entities
    
    enhanced_entities = {}
    
    # Check if entities are in flat format (specialized extractors) or categorized format
    is_flat_format = all(isinstance(v, str) for v in entities.values() if v)
    
    if is_flat_format:
        # Handle flat entity format from specialized extractors
        for field_name, field_value in entities.items():
            enhanced_entities[field_name] = field_value
            
            # Add bounding box if field has a value
            if field_value and isinstance(field_value, str):
                bbox = find_entity_bounding_box(field_value, word_boxes)
                if bbox:
                    # Store bbox info separately to be used later during transformation
                    bbox_field = f"{field_name}_bbox"
                    enhanced_entities[bbox_field] = bbox
    else:
        # Handle categorized entity format  
        for category, entity_list in entities.items():
            enhanced_entities[category] = []
            
            if isinstance(entity_list, list):
                for entity in entity_list:
                    enhanced_entity = entity.copy()
                    
                    # Find bounding box for this entity value
                    if isinstance(entity, dict) and "value" in entity:
                        bbox = find_entity_bounding_box(entity["value"], word_boxes)
                        if bbox:
                            enhanced_entity["bbox"] = bbox
                    
                    enhanced_entities[category].append(enhanced_entity)
            else:
                # Handle non-list entities
                enhanced_entities[category] = entity_list
    
    return enhanced_entities


def find_entity_bounding_box(entity_value: str, 
                           word_boxes: List[Dict]) -> Optional[Dict]:
    """Find bounding box coordinates for an entity value"""
    if not entity_value or not word_boxes:
        return None
    
    # Clean entity value for matching
    entity_clean = entity_value.strip().lower()
    entity_words = entity_clean.split()
    
    if not entity_words:
        return None
    
    # Try to find exact match first
    for i, word_box in enumerate(word_boxes):
        box_text = word_box["text"].strip().lower()
        
        # Exact match
        if box_text == entity_clean:
            return word_box["bbox"]
        
        # Multi-word entity matching
        if len(entity_words) > 1:
            # Check if this word starts the entity
            if box_text == entity_words[0]:
                # Look for consecutive matching words
                matched_boxes = [word_box]
                j = i + 1
                word_idx = 1
                
                while (j < len(word_boxes) and word_idx < len(entity_words)):
                    next_box = word_boxes[j]
                    next_text = next_box["text"].strip().lower()
                    
                    if next_text == entity_words[word_idx]:
                        matched_boxes.append(next_box)
                        word_idx += 1
                        j += 1
                    else:
                        break
                
                # If we matched all words, combine bounding boxes
                if word_idx == len(entity_words):
                    return combine_bounding_boxes(matched_boxes)
    
    # Fuzzy matching for partial matches
    for word_box in word_boxes:
        box_text = word_box["text"].strip().lower()
        if entity_clean in box_text or box_text in entity_clean:
            return word_box["bbox"]
    
    return None


def combine_bounding_boxes(boxes: List[Dict]) -> Dict:
    """Combine multiple bounding boxes into one"""
    if not boxes:
        return None
    
    if len(boxes) == 1:
        return boxes[0]["bbox"]
    
    # Find the bounding rectangle that encompasses all boxes
    min_x = min(box["bbox"]["x"] for box in boxes)
    min_y = min(box["bbox"]["y"] for box in boxes)
    max_x = max(box["bbox"]["x"] + box["bbox"]["width"] for box in boxes)
    max_y = max(box["bbox"]["y"] + box["bbox"]["height"] for box in boxes)
    
    return {
        "x": min_x,
        "y": min_y,
        "width": max_x - min_x,
        "height": max_y - min_y
    }

async def classify_document_type(text: str) -> str:
    """Auto-classify document type based on content"""
    text_lower = text.lower()
    
    # Define keywords for each document type
    type_keywords = {
        "purchase_order": ["purchase order", "p.o.", "po number", "buyer", "seller", "delivery terms"],
        "invoice": ["invoice", "invoice number", "bill to", "total amount", "payment due"],
        "proforma_invoice": ["proforma", "pro forma", "proforma invoice", "incoterm", "country of origin"],
        "bank_guarantee": ["bank guarantee", "guarantee amount", "beneficiary", "validity period"],
        "lc_application": ["letter of credit", "lc application", "applicant", "lc amount"]
    }
    
    scores = {}
    for doc_type, keywords in type_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        scores[doc_type] = score
    
    # Return type with highest score
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    
    return "unknown"

async def extract_generic_entities(text: str, document_type: str, 
                                 extract_tables: bool) -> Dict[str, Any]:
    """Generic entity extraction using regex patterns"""
    entities = {}
    
    # Common patterns
    patterns = {
        "date": r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
        "amount": r'[$₹€£]\s*[\d,]+\.?\d*',
        "number": r'\b[A-Z0-9]{3,}-?[A-Z0-9]+\b',
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    }
    
    # Extract common entities
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            entities[f"{pattern_name}s"] = matches
    
    # Document-specific extraction
    if document_type == "purchase_order":
        # Extract PO number
        po_match = re.search(r'P\.?O\.?\s*(?:Number|No\.?|#)?\s*[:]\s*([A-Z0-9-]+)', text, re.IGNORECASE)
        if po_match:
            entities["po_number"] = po_match.group(1)
        
        # Extract buyer/seller
        buyer_match = re.search(r'Buyer\s*[:]\s*([^\n]+)', text, re.IGNORECASE)
        if buyer_match:
            entities["buyer_name"] = buyer_match.group(1).strip()
        
        seller_match = re.search(r'Seller\s*[:]\s*([^\n]+)', text, re.IGNORECASE)
        if seller_match:
            entities["seller_name"] = seller_match.group(1).strip()
    
    elif document_type == "invoice":
        # Extract invoice number
        inv_match = re.search(r'Invoice\s*(?:Number|No\.?|#)?\s*[:]\s*([A-Z0-9-]+)', text, re.IGNORECASE)
        if inv_match:
            entities["invoice_number"] = inv_match.group(1)
    
    # Extract tables if requested
    if extract_tables:
        tables = extract_tables_from_text(text)
        if tables:
            entities["tables"] = tables
    
    return entities

def extract_tables_from_text(text: str) -> List[List[List[str]]]:
    """Extract tabular data from text"""
    tables = []
    lines = text.split('\n')
    
    # Simple table detection - looks for lines with multiple columns
    current_table = []
    for line in lines:
        # Check if line has multiple fields separated by spaces/tabs
        fields = re.split(r'\s{2,}|\t', line.strip())
        if len(fields) >= 2:
            current_table.append(fields)
        elif current_table and len(current_table) > 1:
            # End of table
            tables.append(current_table)
            current_table = []
    
    if current_table and len(current_table) > 1:
        tables.append(current_table)
    
    return tables

def validate_field(field_name: str, value: Any, document_type: str) -> Dict[str, Any]:
    """Validate specific field format and content"""
    validation = {"is_valid": True, "message": "Valid", "confidence": 1.0}
    
    # Date fields
    if "date" in field_name.lower():
        date_pattern = r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$'
        if not re.match(date_pattern, str(value)):
            validation["is_valid"] = False
            validation["message"] = "Invalid date format"
            validation["confidence"] = 0.3
    
    # Number fields (PO number, invoice number, etc.)
    elif "number" in field_name.lower():
        if not re.match(r'^[A-Z0-9-]+$', str(value), re.IGNORECASE):
            validation["is_valid"] = False
            validation["message"] = "Invalid number format"
            validation["confidence"] = 0.5
    
    # Amount fields
    elif "amount" in field_name.lower():
        amount_pattern = r'^[$₹€£]?\s*[\d,]+\.?\d*$'
        if not re.match(amount_pattern, str(value)):
            validation["is_valid"] = False
            validation["message"] = "Invalid amount format"
            validation["confidence"] = 0.4
    
    # Email fields
    elif "email" in field_name.lower():
        email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
        if not re.match(email_pattern, str(value)):
            validation["is_valid"] = False
            validation["message"] = "Invalid email format"
            validation["confidence"] = 0.2
    
    return validation

def add_bounding_boxes_to_entities(entities: Dict[str, Any], word_boxes: List[Dict]) -> Dict[str, Any]:
    """
    Add bounding box coordinates to entities based on text matching
    """
    entities_with_boxes = {}
    
    for entity_type, entity_value in entities.items():
        if not entity_value or not isinstance(entity_value, str):
            entities_with_boxes[entity_type] = {
                "value": entity_value,
                "bounding_boxes": [],
                "confidence": 0.0
            }
            continue
        
        # Find matching words/phrases in word_boxes (simplified approach)
        matching_boxes = []
        entity_clean = str(entity_value).lower().strip()
        
        # Try exact match first
        for word_box in word_boxes:
            word_text = word_box.get("text", "").lower().strip()
            
            # Exact match
            if entity_clean == word_text:
                matching_boxes.append({
                    "x1": word_box.get("x1", 0),
                    "y1": word_box.get("y1", 0),
                    "x2": word_box.get("x2", 0),
                    "y2": word_box.get("y2", 0),
                    "page": word_box.get("page", 1),
                    "confidence": word_box.get("confidence", 0.0),
                    "text": word_box.get("text", "")
                })
        
        # If no exact match, try substring matching (entity contained in text)
        if not matching_boxes:
            for word_box in word_boxes:
                word_text = word_box.get("text", "").lower().strip()
                
                # Entity is part of the word_text
                if entity_clean in word_text and len(word_text) <= len(entity_clean) * 3:
                    matching_boxes.append({
                        "x1": word_box.get("x1", 0),
                        "y1": word_box.get("y1", 0),
                        "x2": word_box.get("x2", 0),
                        "y2": word_box.get("y2", 0),
                        "page": word_box.get("page", 1),
                        "confidence": word_box.get("confidence", 0.0),
                        "text": word_box.get("text", "")
                    })
                    
        # If still no matches, try word-by-word matching
        if not matching_boxes:
            entity_words = entity_clean.split()
            for word in entity_words:
                if len(word) > 2:  # Skip short words
                    for word_box in word_boxes:
                        word_text = word_box.get("text", "").lower().strip()
                        if word in word_text:
                            matching_boxes.append({
                                "x1": word_box.get("x1", 0),
                                "y1": word_box.get("y1", 0),
                                "x2": word_box.get("x2", 0),
                                "y2": word_box.get("y2", 0),
                                "page": word_box.get("page", 1),
                                "confidence": word_box.get("confidence", 0.0),
                                "text": word_box.get("text", "")
                            })
        
        entities_with_boxes[entity_type] = {
            "value": entity_value,
            "bounding_boxes": matching_boxes,
            "confidence": len(matching_boxes) if matching_boxes else 0.0
        }
    
    return entities_with_boxes


def calculate_extraction_confidence(entities: Dict[str, Any]) -> float:
    """Calculate confidence score for extraction"""
    if not entities:
        return 0.0
    
    # Simple confidence calculation based on number of extracted fields
    field_count = len(entities)
    non_empty_count = sum(1 for v in entities.values() if v and str(v).strip())
    
    if field_count == 0:
        return 0.0
    
    base_confidence = non_empty_count / field_count
    
    # Bonus for specific high-value fields
    key_fields = ["po_number", "invoice_number", "total_amount", "date"]
    key_field_bonus = sum(0.1 for field in key_fields if field in entities and entities[field])
    
    confidence = min(1.0, base_confidence + key_field_bonus)
    return round(confidence, 2)

def calculate_completeness_score(entities: Dict[str, Any], 
                               expected_fields: List[str]) -> float:
    """Calculate how complete the extraction is"""
    if not expected_fields:
        return 1.0
    
    present_fields = sum(1 for field in expected_fields 
                        if field in entities and entities[field])
    
    return round(present_fields / len(expected_fields), 2)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
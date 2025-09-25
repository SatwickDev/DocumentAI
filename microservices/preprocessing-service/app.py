"""
Preprocessing Microservice
Provides adaptive preprocessing operations for document images
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import os
import tempfile
import logging
from typing import Optional, List, Dict, Any
import base64
from datetime import datetime
import sys
import time

# Add pre_processing_updated directory to path for new preprocessing component  
pre_processing_path = os.path.join(os.path.dirname(__file__), '..', '..', 'pre_processing_updated')
sys.path.append(pre_processing_path)
from preprocessing_ops import adaptive_preprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Preprocessing Service",
    description="Adaptive document preprocessing service",
    version="1.0.0"
)

# Service configuration
SERVICE_CONFIG = {
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "supported_formats": [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"]
}

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("🚀 Preprocessing Service starting up...")
    logger.info(f"Supported formats: {SERVICE_CONFIG['supported_formats']}")
    logger.info(f"Max file size: {SERVICE_CONFIG['max_file_size'] / 1024 / 1024}MB")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "preprocessing-service",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/config")
async def get_config():
    """Get service configuration"""
    return SERVICE_CONFIG

@app.post("/preprocess")
async def preprocess_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks,
    return_format: str = Form("file")  # "file" or "base64"
):
    """
    Preprocess a document with adaptive operations
    
    Uses intelligent adaptive preprocessing that automatically determines
    and applies the optimal processing operations based on image quality analysis.
    """
    temp_input_path = None
    temp_output_path = None
    
    try:
        # Validate file
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > SERVICE_CONFIG["max_file_size"]:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {SERVICE_CONFIG['max_file_size'] / 1024 / 1024}MB"
            )
        
        # Check file format
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in SERVICE_CONFIG["supported_formats"]:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file format. Supported: {SERVICE_CONFIG['supported_formats']}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(content)
            temp_input_path = tmp_file.name
        
        # Process the document using adaptive preprocessing
        if file_extension == ".pdf":
            # Handle PDF preprocessing
            processed_images = await preprocess_pdf_adaptive(temp_input_path)
            
            # Create output PDF in pre_processing_updated folder
            output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 
                                    'pre_processing_updated', 'processed_documents')
            os.makedirs(output_dir, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(file.filename)[0]
            output_filename = f"{base_name}_preprocessed_{timestamp}.pdf"
            temp_output_path = os.path.join(output_dir, output_filename)
            create_pdf_from_images(processed_images, temp_output_path)
        else:
            # Handle image preprocessing
            img = cv2.imread(temp_input_path)
            if img is None:
                raise HTTPException(status_code=400, detail="Failed to read image")
            
            # Convert to grayscale as expected by adaptive_preprocess
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            
            # Apply adaptive preprocessing and track operations
            processed, deskewed = adaptive_preprocess(img_gray)
            
            # Analyze what operations were likely applied based on image characteristics
            operations_applied = []
            improvements = []
            
            # Calculate image characteristics to determine what operations were applied
            original_contrast = img_gray.std() / 255.0
            original_brightness = np.mean(img_gray) / 255.0
            processed_contrast = processed.std() / 255.0
            processed_brightness = np.mean(processed) / 255.0
            
            # Determine operations based on adaptive_preprocess logic
            if deskewed:
                operations_applied.append("Deskewing")
                improvements.append("Corrected document rotation")
            
            if original_contrast < 0.20:
                operations_applied.append("CLAHE Enhancement")
                improvements.append("Improved local contrast")
            
            if original_contrast < 0.13:
                operations_applied.append("Contrast Stretching")
                improvements.append("Enhanced overall contrast")
            
            # These operations are always applied in adaptive_preprocess
            operations_applied.append("Noise Reduction")
            operations_applied.append("Black Point Enhancement")
            improvements.append("Reduced image noise")
            improvements.append("Enhanced text clarity")
            
            if abs(processed_brightness - original_brightness) > 0.05:
                operations_applied.append("Brightness Normalization")
                improvements.append("Optimized brightness levels")
            
            # Save processed image to pre_processing_updated folder
            output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 
                                    'pre_processing_updated', 'processed_documents')
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(file.filename)[0]
            output_filename = f"{base_name}_preprocessed_{timestamp}{file_extension}"
            temp_output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(temp_output_path, processed)
        
        # Return result based on format
        if return_format == "base64":
            with open(temp_output_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            
            # Clean up files
            background_tasks.add_task(cleanup_files, [temp_input_path, temp_output_path])
            
            return {
                "status": "success",
                "operations_applied": operations_applied,
                "improvements": improvements,
                "format": "base64",
                "data": encoded,
                "original_filename": file.filename,
                "processed_filename": os.path.basename(temp_output_path),
                "image_analysis": {
                    "original_contrast": round(original_contrast, 3),
                    "original_brightness": round(original_brightness, 3),
                    "processed_brightness": round(processed_brightness, 3),
                    "deskewed": deskewed
                }
            }
        else:
            # Only cleanup the temporary input file, keep the processed output
            background_tasks.add_task(cleanup_file_delayed, temp_input_path, 60)
            
            return FileResponse(
                temp_output_path,
                media_type=f"application/{file_extension[1:]}",
                filename=f"preprocessed_{file.filename}"
            )
    
    except Exception as e:
        logger.error(f"Error preprocessing document: {e}")
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if temp_output_path and os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        
        if isinstance(e, HTTPException):
            raise e
        
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@app.post("/preprocess/batch")
async def preprocess_batch(
    files: List[UploadFile] = File(...)
):
    """Preprocess multiple documents using adaptive preprocessing"""
    results = []
    
    for file in files:
        try:
            # Process each file using adaptive preprocessing
            result = await preprocess_document(
                file=file,
                background_tasks=BackgroundTasks(),
                return_format="base64"
            )
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "data": result["data"]
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}

@app.post("/preprocess/selective")
async def preprocess_selective(
    file: UploadFile = File(...),
    quality_results: str = Form(...),  # JSON string with page-level quality results
    background_tasks: BackgroundTasks = BackgroundTasks
):
    """
    Selective preprocessing: Only process pages that need preprocessing
    Creates a mixed PDF with original and processed pages with _processed suffix
    """
    temp_input_path = None
    temp_output_path = None
    
    try:
        # Parse quality results
        import json
        try:
            quality_data = json.loads(quality_results)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid quality_results JSON: {e}")
        
        # Validate file
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        content = await file.read()
        
        # Check file format - only support PDF for selective processing
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension != ".pdf":
            raise HTTPException(
                status_code=415,
                detail="Selective preprocessing only supports PDF files"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(content)
            temp_input_path = tmp_file.name
        
        # Process PDF with selective preprocessing
        temp_output_path = await preprocess_pdf_selective(temp_input_path, quality_data, file.filename)
        
        # Only cleanup the temporary input file, keep the processed output
        background_tasks.add_task(cleanup_file_delayed, temp_input_path, 60)
        
        return FileResponse(
            temp_output_path,
            media_type="application/pdf",
            filename=f"{os.path.splitext(file.filename)[0]}_processed.pdf"
        )
    
    except Exception as e:
        logger.error(f"Error in selective preprocessing: {e}")
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if temp_output_path and os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        
        if isinstance(e, HTTPException):
            raise e
        
        raise HTTPException(status_code=500, detail=f"Selective preprocessing failed: {str(e)}")

@app.post("/analyze-preprocessing-needs")
async def analyze_preprocessing_needs(
    file: UploadFile = File(...),
):
    """Analyze document - now uses adaptive preprocessing logic"""
    return {
        "filename": file.filename,
        "analysis": "Uses adaptive preprocessing - automatically determines optimal operations",
        "recommended_operations": ["adaptive_preprocessing"],
        "needs_preprocessing": True
    }

# Selective preprocessing functions

async def preprocess_pdf_selective(pdf_path: str, quality_data: Dict[str, Any],
                                   original_filename: str) -> str:
    """
    Selective preprocessing: Only process pages that need preprocessing
    Creates a mixed PDF with original and processed pages
    """
    import fitz
    
    # Extract page results from quality data
    page_results = quality_data.get('page_results', [])
    if not page_results:
        raise ValueError("No page results found in quality data")
    
    # Open original PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # Create new document for output
    output_doc = fitz.open()
    
    pages_processed = []
    processing_info = {
        "total_pages": total_pages,
        "pages_processed": [],
        "pages_skipped": [],
        "processing_time": 0
    }
    
    start_time = time.time()
    
    for page_num in range(total_pages):
        try:
            # Check if this page needs preprocessing
            page_quality = None
            if page_num < len(page_results):
                page_quality = page_results[page_num]
            
            needs_processing = False
            if page_quality:
                # Check preprocessing decision from API Gateway
                preprocessing = page_quality.get('preprocessing', {})
                needs_processing = preprocessing.get('needs_preprocessing',
                                                     False)
                
                # Fallback to verdict-based logic if no preprocessing field
                if not preprocessing:
                    verdict = page_quality.get('verdict', '').lower()
                    needs_processing = verdict in ['pre-processing',
                                                   'azure document analysis']
            
            original_page = doc[page_num]
            
            if needs_processing:
                # Extract page as image
                pix = original_page.get_pixmap(dpi=200)
                img_data = pix.tobytes("png")
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8),
                                   cv2.IMREAD_COLOR)
                
                # Convert to grayscale as expected by adaptive_preprocess
                if len(img.shape) == 3:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    img_gray = img
                
                # Apply adaptive preprocessing
                processed, deskewed = adaptive_preprocess(img_gray)
                
                # Convert processed image back to PDF page
                processed_pil = Image.fromarray(processed)
                img_bytes = io.BytesIO()
                processed_pil.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                # Create new page with processed image
                page_rect = original_page.rect
                new_page = output_doc.new_page(width=page_rect.width,
                                               height=page_rect.height)
                new_page.insert_image(page_rect, stream=img_bytes.getvalue())
                
                pages_processed.append(page_num + 1)
                verdict = (page_quality.get('verdict', '')
                           if page_quality else '')
                score = (page_quality.get('score', 0.0)
                         if page_quality else 0.0)
                processing_info["pages_processed"].append({
                    "page": page_num + 1,
                    "verdict": verdict,
                    "score": score,
                    "operations": ["adaptive_preprocessing"],
                    "deskewed": deskewed
                })
                
                logger.info(f"Processed page {page_num + 1} with "
                            f"adaptive preprocessing")
            else:
                # Use original page without processing
                output_doc.insert_pdf(doc, from_page=page_num,
                                       to_page=page_num)
                verdict = (page_quality.get('verdict', '')
                           if page_quality else '')
                score = (page_quality.get('score', 0.0)
                         if page_quality else 0.0)
                processing_info["pages_skipped"].append({
                    "page": page_num + 1,
                    "reason": "Quality sufficient for direct analysis",
                    "verdict": verdict,
                    "score": score
                })
                
                logger.info(f"Kept original page {page_num + 1} "
                            f"(no processing needed)")
                
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")
            # Fall back to original page
            output_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
    
    processing_info["processing_time"] = time.time() - start_time
    
    # Save output PDF to pre_processing_updated folder
    base_name = os.path.splitext(original_filename)[0]
    # Create pre_processing_updated folder if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'pre_processing_updated', 'processed_documents')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp for uniqueness
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_name}_processed_{timestamp}.pdf"
    output_path = os.path.join(output_dir, output_filename)
    output_doc.save(output_path)
    
    # Close documents
    doc.close()
    output_doc.close()
    
    logger.info(f"Selective preprocessing complete: "
                f"{len(pages_processed)} pages processed, "
                f"{total_pages - len(pages_processed)} pages kept original")
    
    return output_path


# Old manual preprocessing functions removed - now using adaptive preprocessing only

async def preprocess_pdf_adaptive(pdf_path: str) -> List[np.ndarray]:
    """Preprocess all pages in a PDF using adaptive preprocessing"""
    import fitz
    
    doc = fitz.open(pdf_path)
    processed_images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        img_data = pix.tobytes("png")
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Convert to grayscale as expected by adaptive_preprocess
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        
        # Apply adaptive preprocessing using your new component
        processed, deskewed = adaptive_preprocess(img_gray)
        processed_images.append(processed)
    
    doc.close()
    return processed_images

def create_pdf_from_images(images: List[np.ndarray], output_path: str):
    """Create PDF from preprocessed images"""
    import fitz
    
    doc = fitz.open()
    
    for img in images:
        # Convert numpy array to PIL Image
        if len(img.shape) == 3:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.fromarray(img)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Add to PDF
        img_doc = fitz.open(stream=img_bytes.read(), filetype="png")
        pdf_page = doc.new_page(width=img_doc[0].rect.width, 
                               height=img_doc[0].rect.height)
        pdf_page.insert_image(pdf_page.rect, stream=img_bytes.getvalue())
        img_doc.close()
    
    doc.save(output_path)
    doc.close()

def cleanup_files(file_paths: List[str]):
    """Clean up temporary files"""
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.error(f"Error cleaning up file {path}: {e}")

def cleanup_file_delayed(file_path: str, delay_seconds: int):
    """Clean up file after delay"""
    import time
    time.sleep(delay_seconds)
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
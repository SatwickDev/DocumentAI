# Entity Extraction Service - Request & Response Documentation

## Service Details
- **Port:** 8004
- **Base URL:** http://localhost:8004
- **Service:** Entity Extraction

---

## Entity Extraction Request & Response

### POST `/extract` - Extract Entities from Document

**Request:**
```bash
curl -X POST "http://localhost:8004/extract" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "document_type=purchase_order" \
  -F "confidence_threshold=0.6" \
  -F "extract_tables=true"
```

**Request Parameters:**
- `file` (required): PDF/Image file to extract entities from
- `document_type` (optional): Type of document ("purchase_order", "proforma_invoice", "invoice", "bank_guarantee")
- `confidence_threshold` (optional): Minimum confidence for extraction (default: 0.6)
- `extract_tables` (optional): Whether to extract tabular data (default: true)

---

### **ACTUAL Response Format** (Based on your entity extraction service):

```json
{
  "entities": {
    "po_number": {
      "value": "PO/UAE/2025/001",
      "confidence": 0.95,
      "location": {
        "page": 1,
        "bbox": [120, 180, 250, 200]
      }
    },
    "po_date": {
      "value": "25.09.2025",
      "confidence": 0.92,
      "location": {
        "page": 1,
        "bbox": [350, 180, 450, 200]
      }
    },
    "seller_name": {
      "value": "ABC Trading LLC, Dubai, United Arab Emirates",
      "confidence": 0.88,
      "location": {
        "page": 1,
        "bbox": [100, 220, 400, 280]
      }
    },
    "buyer_name": {
      "value": "XYZ Corporation, Business Bay, Dubai, UNITED ARAB EMIRATES",
      "confidence": 0.91,
      "location": {
        "page": 1,
        "bbox": [100, 300, 400, 360]
      }
    },
    "goods_description": {
      "value": "Office Furniture: 10 x Executive Desks, 20 x Office Chairs, 5 x Conference Tables",
      "confidence": 0.85,
      "location": {
        "page": 1,
        "bbox": [100, 400, 500, 480]
      }
    },
    "total_amount": {
      "value": "AED 125,000.00",
      "confidence": 0.93,
      "location": {
        "page": 1,
        "bbox": [400, 550, 500, 570]
      }
    },
    "delivery_terms": {
      "value": "FOB Dubai Port",
      "confidence": 0.78,
      "location": {
        "page": 1,
        "bbox": [100, 600, 300, 620]
      }
    },
    "payment_terms": {
      "value": "30 days from invoice date",
      "confidence": 0.82,
      "location": {
        "page": 1,
        "bbox": [100, 640, 350, 660]
      }
    },
    "validity_period": {
      "value": "Valid until 31.12.2025",
      "confidence": 0.87,
      "location": {
        "page": 1,
        "bbox": [100, 680, 300, 700]
      }
    }
  },
  "processed_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA...",
  "image_dimensions": {
    "width": 800,
    "height": 1200
  }
}
```

---

## Multi-PDF Entity Extraction

### POST `/extract/multi-pdf` - Extract from Classified Document Categories

**Request:**
```bash
curl -X POST "http://localhost:8004/extract/multi-pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "session_id=12345678-1234-1234-1234-123456789012" \
  -F "document_name=sample_document" \
  -F "pdf_summary={\"Purchase Order\": {...}, \"Invoice\": {...}}" \
  -F "output_directory=/app/document_classification_updated/sample_document"
```

**Response:**
```json
{
  "session_id": "12345678-1234-1234-1234-123456789012",
  "document_name": "sample_document",
  "total_categories": 2,
  "total_pdfs_processed": 3,
  "processing_errors": [],
  "categories": {
    "Purchase Order": {
      "pdf_files": [
        {
          "filename": "sample_document_page_1_Purchase_Order.pdf",
          "confidence": 0.89,
          "entities": {
            "po_number": "PO/UAE/2025/001",
            "seller_name": "ABC Trading LLC",
            "total_amount": "AED 125,000.00"
          }
        }
      ],
      "total_entities": 9,
      "avg_confidence": 0.87
    },
    "Invoice": {
      "pdf_files": [
        {
          "filename": "sample_document_page_3_Invoice.pdf",
          "confidence": 0.82,
          "entities": {
            "invoice_number": "INV-2025-100",
            "customer_name": "XYZ Corporation",
            "amount_due": "AED 85,000.00"
          }
        }
      ],
      "total_entities": 7,
      "avg_confidence": 0.81
    }
  }
}
```

---

## Batch Processing

### POST `/extract/batch` - Extract from Multiple Documents

**Request:**
```bash
curl -X POST "http://localhost:8004/extract/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "document_types=[\"purchase_order\", \"invoice\"]" \
  -F "confidence_threshold=0.7"
```

**Response:**
```json
{
  "batch_id": "batch_456789",
  "total_documents": 2,
  "successful": 2,
  "failed": 0,
  "processing_time": 4.67,
  "results": [
    {
      "filename": "doc1.pdf",
      "document_type": "purchase_order",
      "status": "success",
      "confidence": 0.89,
      "entities_count": 9,
      "entities": {
        "po_number": "PO/UAE/2025/002",
        "seller_name": "DEF Trading LLC",
        "total_amount": "AED 75,000.00"
      }
    },
    {
      "filename": "doc2.pdf", 
      "document_type": "invoice",
      "status": "success",
      "confidence": 0.84,
      "entities_count": 7,
      "entities": {
        "invoice_number": "INV-2025-101",
        "customer_name": "ABC Corporation",
        "amount_due": "AED 95,000.00"
      }
    }
  ]
}
```

---

## Supported Document Types & Fields

### Purchase Order
**Fields Extracted:**
- `po_number`: Purchase order number
- `po_date`: Purchase order date
- `seller_name`: Seller/vendor information
- `buyer_name`: Buyer information
- `goods_description`: Description of goods/services
- `total_amount`: Total amount with currency
- `delivery_terms`: Delivery terms (FOB, CIF, etc.)
- `payment_terms`: Payment terms and conditions
- `validity_period`: Validity period of the PO

### Proforma Invoice
**Fields Extracted:**
- `proforma_number`: Proforma invoice number
- `proforma_date`: Date of proforma invoice
- `seller_name`: Seller information
- `buyer_name`: Buyer information
- `goods_description`: Description of goods
- `total_amount`: Total amount
- `payment_terms`: Payment terms
- `validity_date`: Validity date
- `terms_conditions`: Terms and conditions

### Invoice
**Fields Extracted:**
- `invoice_number`: Invoice number
- `invoice_date`: Invoice date
- `customer_name`: Customer information
- `amount_due`: Amount due
- `due_date`: Payment due date
- `line_items`: Individual line items
- `tax_amount`: Tax amount
- `discount`: Discount amount

### Bank Guarantee
**Fields Extracted:**
- `guarantee_number`: Bank guarantee number
- `beneficiary_name`: Beneficiary information
- `guarantee_amount`: Guarantee amount
- `validity_date`: Validity date
- `issuing_bank`: Issuing bank details
- `terms_conditions`: Terms and conditions

---

## Entity Structure

Each extracted entity contains:
```json
{
  "value": "Extracted text value",
  "confidence": 0.95,
  "location": {
    "page": 1,
    "bbox": [x1, y1, x2, y2]
  }
}
```

- **`value`**: The extracted text value
- **`confidence`**: Confidence score (0.0 - 1.0)
- **`location.page`**: Page number where entity was found
- **`location.bbox`**: Bounding box coordinates [x1, y1, x2, y2]

---

## Service Endpoints

### GET `/health` - Health Check

**Request:**
```bash
curl -X GET "http://localhost:8004/health"
```

**Response:**
```json
{
  "status": "healthy",
  "service": "entity-extraction-service",
  "timestamp": "2025-09-25T16:00:00Z",
  "extractors_loaded": ["purchase_order", "proforma_invoice"]
}
```

### GET `/supported-types` - Get Supported Document Types

**Request:**
```bash
curl -X GET "http://localhost:8004/supported-types"
```

**Response:**
```json
{
  "purchase_order": {
    "fields": ["po_number", "po_date", "seller_name", "buyer_name", "goods_description", "total_amount", "delivery_terms", "payment_terms", "validity_period"],
    "description": "Purchase Order documents"
  },
  "proforma_invoice": {
    "fields": ["proforma_number", "proforma_date", "seller_name", "buyer_name", "goods_description", "total_amount", "payment_terms", "validity_date", "terms_conditions"],
    "description": "Proforma Invoice documents"
  }
}
```

### GET `/config` - Get Service Configuration

**Request:**
```bash
curl -X GET "http://localhost:8004/config"
```

**Response:**
```json
{
  "supported_document_types": ["purchase_order", "invoice", "proforma_invoice", "bank_guarantee"],
  "max_file_size": 52428800,
  "ocr_languages": ["eng"],
  "confidence_threshold": 0.6
}
```

---

## Error Response Format

```json
{
  "detail": "Extraction failed: Unsupported document type"
}
```

### Common Error Messages:
- `"No file provided"` (400): No file uploaded
- `"File too large. Max size: 50MB"` (413): File exceeds size limit
- `"Unsupported document type: [type]"` (400): Document type not supported
- `"Extraction failed: [reason]"` (500): Processing error
- `"Service not fully initialized"` (503): Service still starting up

---

## OCR and Text Processing

The service includes:
- **Tesseract OCR** for text extraction from images and scanned PDFs
- **PyMuPDF** for native PDF text extraction
- **OpenCV** for image preprocessing
- **PIL** for image manipulation
- **Pattern matching** using regular expressions
- **Confidence scoring** based on extraction quality

---

## Frontend Integration

The response provides:
1. **Extracted Entities** (`entities`): All extracted field values with confidence scores
2. **Processed Image** (`processed_image`): Base64 encoded image for display
3. **Image Dimensions** (`image_dimensions`): Width and height for proper scaling
4. **Bounding Box Coordinates** (`location.bbox`): For highlighting entities on the image
5. **Confidence Scores** (`confidence`): For displaying extraction quality to users

The frontend can use the bounding box coordinates to highlight extracted entities on the processed image, providing visual feedback to users about what was extracted and where it was found.
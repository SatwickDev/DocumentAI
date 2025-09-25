# Enhanced Document Processing Services

## Overview

This document describes the enhanced microservices and MCP servers that have been added to the F2 document processing system. These services provide advanced quality analysis, preprocessing, and entity extraction capabilities.

## New Services

### 1. Enhanced Quality Analyzer
- **Location**: `src/core/enhanced_quality_analyzer.py`
- **Features**:
  - 16 comprehensive quality metrics (blur, contrast, noise, sharpness, brightness, skew, text coverage, OCR confidence, margins, edge crop, shadow/glare, blank page, resolution, and more)
  - YAML-based configuration
  - Adaptive preprocessing recommendations
  - Parallel processing with ThreadPoolExecutor
  - Numba JIT optimization for performance

### 2. Preprocessing Microservice
- **Port**: 8003
- **Location**: `microservices/preprocessing-service/app.py`
- **Endpoints**:
  - `POST /preprocess` - Apply preprocessing operations
  - `POST /preprocess/batch` - Process multiple documents
  - `POST /analyze-preprocessing-needs` - Analyze what preprocessing is needed
- **Operations**:
  - Deskewing
  - Contrast enhancement (CLAHE)
  - Denoising
  - Brightness normalization
  - Adaptive preprocessing (all)

### 3. Entity Extraction Microservice
- **Port**: 8004
- **Location**: `microservices/entity-extraction-service/app.py`
- **Endpoints**:
  - `POST /extract` - Extract entities from documents
  - `POST /extract/batch` - Extract from multiple documents
  - `POST /validate-extraction` - Validate extracted entities
  - `GET /supported-types` - Get supported document types
- **Supported Documents**:
  - Purchase Orders
  - Invoices
  - Proforma Invoices
  - Bank Guarantees
  - Letter of Credit Applications

### 4. Enhanced API Gateway
- **Port**: 8000
- **Location**: `microservices/api-gateway/enhanced_app.py`
- **New Endpoints**:
  - `POST /analyze/quality` - Quality analysis with preprocessing
  - `POST /preprocess` - Document preprocessing
  - `POST /extract/entities` - Entity extraction
  - `POST /classify/entities` - Classification + entity extraction
  - `POST /process/full-pipeline` - Complete processing pipeline

## MCP Servers

### 1. Enhanced Quality MCP Server
- **Location**: `src/mcp_servers/enhanced_quality_mcp_server.py`
- **Tools**:
  - `analyze_quality_enhanced` - Comprehensive quality analysis
  - `analyze_preprocessing_needs` - Determine preprocessing requirements
  - `get_quality_config` - Get configuration
  - `validate_quality_metrics` - Validate metrics

### 2. Preprocessing MCP Server
- **Location**: `src/mcp_servers/preprocessing_mcp_server.py`
- **Tools**:
  - `preprocess_document` - Apply preprocessing
  - `preprocess_batch` - Batch preprocessing
  - `get_available_operations` - List operations

### 3. Entity Extraction MCP Server
- **Location**: `src/mcp_servers/entity_extraction_mcp_server.py`
- **Tools**:
  - `extract_entities` - Extract structured data
  - `validate_extraction` - Validate results
  - `get_supported_types` - List document types
  - `classify_document` - Auto-classify document type

## MCP Contracts

- **Location**: `contracts/mcp_enhanced_contracts.json`
- **Contents**:
  - JSON Schema definitions for all services
  - Request/Response contracts
  - Tool definitions
  - Validation schemas

## How to Run

### 1. Start All Services

```bash
# Windows
python start_enhanced_services.py

# Linux/Mac
python3 start_enhanced_services.py
```

This will start:
- Enhanced API Gateway (port 8000)
- Classification Service (port 8001)
- Quality Service (port 8002)
- Preprocessing Service (port 8003)
- Entity Extraction Service (port 8004)

### 2. Test the Services

```bash
# Run API tests
python test_api_endpoints.py
```

### 3. Access the API Documentation

Open your browser to: http://localhost:8000/docs

## Example Usage

### Full Pipeline Processing

```python
import requests

# Process a document through the full pipeline
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/process/full-pipeline",
        files={"file": ("document.pdf", f, "application/pdf")}
    )
    
result = response.json()
print(f"Success Rate: {result['success_rate'] * 100}%")
print(f"Classification: {result['pipeline_stages']['classification']}")
print(f"Entities: {result['pipeline_stages']['entity_extraction']}")
```

### Quality Analysis with Preprocessing

```python
# Analyze quality and apply preprocessing if needed
with open("poor_quality.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze/quality",
        files={"file": ("poor_quality.pdf", f, "application/pdf")},
        params={"apply_preprocessing": True}
    )
    
quality_result = response.json()
print(f"Verdict: {quality_result['verdict']}")
print(f"Overall Score: {quality_result['overall_score']}")
```

### Entity Extraction

```python
# Extract entities from a purchase order
with open("purchase_order.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/extract/entities",
        files={"file": ("purchase_order.pdf", f, "application/pdf")},
        params={"document_type": "purchase_order"}
    )
    
entities = response.json()
print(f"PO Number: {entities['entities'].get('po_number')}")
print(f"Total Amount: {entities['entities'].get('total_amount')}")
```

## Configuration

### Quality Configuration

Edit `config/quality_config.yaml` to adjust:
- Quality thresholds
- Metric weights
- Preprocessing triggers

### Classification Configuration

Edit `config/classification_config.json` to:
- Add new document types
- Update keywords
- Adjust weights

## Architecture

```
┌─────────────────────────────────────────────┐
│           Enhanced API Gateway               │
│              (Port 8000)                     │
├─────────────────────────────────────────────┤
│  Routes requests to appropriate services     │
└─────────────┬───────────────────────────────┘
              │
    ┌─────────┴─────────┬─────────────┬─────────────┐
    │                   │             │             │
┌───▼────┐      ┌──────▼──────┐  ┌───▼────┐   ┌───▼────┐
│Quality │      │Preprocessing│  │Entity  │   │Classify│
│Service │      │Service      │  │Extract │   │Service │
│ (8002) │      │   (8003)    │  │ (8004) │   │ (8001) │
└────────┘      └─────────────┘  └────────┘   └────────┘
```

## Development

### Adding New Quality Metrics

1. Add metric to `EnhancedQualityMetrics` dataclass
2. Implement calculation in `_calculate_all_metrics()`
3. Add threshold configuration
4. Update contract definitions

### Adding New Document Types

1. Create extractor in `src/temp/`
2. Register in `EntityExtractionMCPServer`
3. Add to supported types
4. Update contracts

### Adding New Preprocessing Operations

1. Implement in `preprocessing_ops.py`
2. Add to `apply_selected_operations()`
3. Update MCP contracts
4. Add to API documentation

## Troubleshooting

### NumPy Compatibility Issues

If you encounter NumPy version conflicts:
```bash
pip install "numpy<2.0"
```

### Port Already in Use

Check which process is using the port:
```bash
# Windows
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000
```

### Service Not Starting

Check logs in the console output or verify dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-mcp.txt
```

## Performance Tips

1. **Batch Processing**: Use batch endpoints for multiple documents
2. **Preprocessing Cache**: Preprocessed documents can be saved for reuse
3. **Parallel Processing**: The quality analyzer uses parallel processing automatically
4. **Configuration Tuning**: Adjust thresholds based on your document types

## Security Considerations

1. **File Size Limits**: Default 100MB for quality/preprocessing, 50MB for extraction
2. **Timeout Settings**: 60-second timeout for most operations
3. **Input Validation**: All inputs are validated before processing
4. **Error Handling**: Sensitive information is not exposed in error messages

## Future Enhancements

1. **Machine Learning Integration**: Replace keyword-based classification
2. **Multi-Language Support**: Add language detection and multi-lingual extraction
3. **Distributed Processing**: Add support for distributed task queues
4. **Real-time Monitoring**: Add metrics and monitoring dashboards
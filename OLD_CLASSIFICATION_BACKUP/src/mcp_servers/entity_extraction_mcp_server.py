"""
Entity Extraction MCP Server
Provides entity extraction from business documents via MCP protocol
"""
import asyncio
import json
import base64
import os
import sys
import tempfile
import re
from typing import Dict, Any, List, Optional
import logging

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'temp'))
from mcp.core import MCPServer, MCPRequest, MCPResponse, MCPError, MCPContext
from purchase_order import PurchaseOrderExtractor
from performa_invoice import PerformaInvoiceExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityExtractionMCPServer(MCPServer):
    """MCP Server for entity extraction from documents"""
    
    def __init__(self):
        super().__init__("entity-extraction-server")
        
        # Initialize extractors
        self.extractors = {
            "purchase_order": PurchaseOrderExtractor(),
            "proforma_invoice": PerformaInvoiceExtractor(),
            "performa_invoice": PerformaInvoiceExtractor(),  # Alias
        }
        
        self.supported_types = [
            "purchase_order",
            "invoice", 
            "proforma_invoice",
            "bank_guarantee",
            "lc_application"
        ]
        
        self.register_handlers()
        
    def register_handlers(self):
        """Register all MCP method handlers"""
        # Register tools
        self.register_method("tools/list", self.handle_tools_list)
        
        # Register extraction methods
        self.register_method("extract_entities", self.handle_extract_entities)
        self.register_method("validate_extraction", self.handle_validate_extraction)
        self.register_method("get_supported_types", self.handle_get_supported_types)
        self.register_method("classify_document", self.handle_classify_document)
        
    async def handle_tools_list(self, request: MCPRequest, context: MCPContext) -> MCPResponse:
        """List available tools"""
        tools = [
            {
                "name": "extract_entities",
                "description": "Extract structured entities from business documents",
                "inputSchema": {
                    "type": "object",
                    "required": ["document"],
                    "properties": {
                        "document": {
                            "type": "string",
                            "format": "base64",
                            "description": "Base64 encoded document"
                        },
                        "document_type": {
                            "type": "string",
                            "enum": self.supported_types + ["auto"],
                            "default": "auto"
                        },
                        "confidence_threshold": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.6
                        },
                        "extract_tables": {
                            "type": "boolean",
                            "default": True
                        }
                    }
                }
            },
            {
                "name": "validate_extraction",
                "description": "Validate extracted entities against expected schema",
                "inputSchema": {
                    "type": "object",
                    "required": ["entities", "document_type"],
                    "properties": {
                        "entities": {
                            "type": "object"
                        },
                        "document_type": {
                            "type": "string"
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
    
    async def handle_extract_entities(self, request: MCPRequest, context: MCPContext) -> MCPResponse:
        """Handle entity extraction request"""
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
            
            # Get parameters
            document_type = params.get("document_type", "auto")
            confidence_threshold = params.get("confidence_threshold", 0.6)
            extract_tables = params.get("extract_tables", True)
            
            # Save to temporary file
            suffix = '.pdf' if document_data[:4] == b'%PDF' else '.png'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(document_data)
                temp_path = tmp_file.name
            
            # Extract text from document
            extracted_text = await self._extract_text_from_file(temp_path)
            
            # Determine document type if auto
            if document_type == "auto":
                document_type = await self._classify_document_type(extracted_text)
                logger.info(f"Auto-detected document type: {document_type}")
            
            # Validate document type
            if document_type not in self.supported_types:
                return MCPResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error=MCPError(
                        code=-32602,
                        message="Invalid params",
                        data=f"Unsupported document type: {document_type}"
                    )
                )
            
            # Extract entities
            if document_type in self.extractors:
                # Use specialized extractor
                extractor = self.extractors[document_type]
                entities = extractor.extract_from_file(temp_path)
            else:
                # Use generic extraction
                entities = await self._extract_generic_entities(
                    extracted_text,
                    document_type,
                    extract_tables
                )
            
            # Calculate confidence
            confidence = self._calculate_confidence(entities)
            
            # Apply threshold
            if confidence < confidence_threshold:
                logger.warning(f"Low confidence extraction: {confidence} < {confidence_threshold}")
            
            response_data = {
                "status": "success",
                "document_type": document_type,
                "entities": entities,
                "confidence": confidence,
                "text_length": len(extracted_text),
                "extraction_method": "specialized" if document_type in self.extractors else "generic"
            }
            
            return MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                result=response_data
            )
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
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
    
    async def handle_validate_extraction(self, request: MCPRequest, context: MCPContext) -> MCPResponse:
        """Validate extracted entities"""
        try:
            params = request.params
            
            if "entities" not in params or "document_type" not in params:
                return MCPResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error=MCPError(
                        code=-32602,
                        message="Invalid params",
                        data="Missing required parameters"
                    )
                )
            
            entities = params["entities"]
            document_type = params["document_type"]
            
            # Get expected fields
            expected_fields = self._get_expected_fields(document_type)
            
            # Validate
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
                    # Validate field format
                    validation = self._validate_field(field, entities[field], document_type)
                    validation_result["field_validation"][field] = validation
                    if not validation["is_valid"]:
                        validation_result["is_valid"] = False
            
            validation_result["completeness_score"] = self._calculate_completeness(
                entities, expected_fields
            )
            
            return MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                result=validation_result
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
    
    async def handle_get_supported_types(self, request: MCPRequest, context: MCPContext) -> MCPResponse:
        """Get supported document types"""
        supported = {}
        
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
        
        return MCPResponse(
            jsonrpc="2.0",
            id=request.id,
            result={"supported_types": supported}
        )
    
    async def handle_classify_document(self, request: MCPRequest, context: MCPContext) -> MCPResponse:
        """Classify document type from text"""
        try:
            params = request.params
            
            if "text" not in params:
                return MCPResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error=MCPError(
                        code=-32602,
                        message="Invalid params",
                        data="Missing required parameter: text"
                    )
                )
            
            text = params["text"]
            document_type = await self._classify_document_type(text)
            
            return MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                result={
                    "document_type": document_type,
                    "confidence": 0.8 if document_type != "unknown" else 0.0
                }
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
    
    async def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from file"""
        if file_path.lower().endswith('.pdf'):
            import fitz
            doc = fitz.open(file_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                if len(page_text.strip()) < 50:
                    # Use OCR if no text layer
                    import pytesseract
                    import cv2
                    import numpy as np
                    
                    pix = page.get_pixmap(dpi=200)
                    img_data = pix.tobytes("png")
                    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                    page_text = pytesseract.image_to_string(img)
                
                text += page_text + "\n"
            
            doc.close()
            return text
        else:
            # Handle images
            import pytesseract
            import cv2
            
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Failed to read image file")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            return text
    
    async def _classify_document_type(self, text: str) -> str:
        """Auto-classify document type"""
        text_lower = text.lower()
        
        type_keywords = {
            "purchase_order": ["purchase order", "p.o.", "po number", "buyer", "seller"],
            "invoice": ["invoice", "invoice number", "bill to", "total amount"],
            "proforma_invoice": ["proforma", "pro forma", "proforma invoice", "incoterm"],
            "bank_guarantee": ["bank guarantee", "guarantee amount", "beneficiary"],
            "lc_application": ["letter of credit", "lc application", "applicant"]
        }
        
        scores = {}
        for doc_type, keywords in type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[doc_type] = score
        
        if scores:
            best_match = max(scores.items(), key=lambda x: x[1])
            if best_match[1] > 0:
                return best_match[0]
        
        return "unknown"
    
    async def _extract_generic_entities(self, text: str, document_type: str,
                                      extract_tables: bool) -> Dict[str, Any]:
        """Generic entity extraction"""
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
            po_match = re.search(r'P\.?O\.?\s*(?:Number|No\.?|#)?\s*[:]\s*([A-Z0-9-]+)', 
                               text, re.IGNORECASE)
            if po_match:
                entities["po_number"] = po_match.group(1)
            
            buyer_match = re.search(r'Buyer\s*[:]\s*([^\n]+)', text, re.IGNORECASE)
            if buyer_match:
                entities["buyer_name"] = buyer_match.group(1).strip()
        
        elif document_type == "invoice":
            inv_match = re.search(r'Invoice\s*(?:Number|No\.?|#)?\s*[:]\s*([A-Z0-9-]+)',
                                text, re.IGNORECASE)
            if inv_match:
                entities["invoice_number"] = inv_match.group(1)
        
        # Extract tables if requested
        if extract_tables:
            tables = self._extract_tables(text)
            if tables:
                entities["tables"] = tables
        
        return entities
    
    def _extract_tables(self, text: str) -> List[List[List[str]]]:
        """Extract tabular data"""
        tables = []
        lines = text.split('\n')
        
        current_table = []
        for line in lines:
            fields = re.split(r'\s{2,}|\t', line.strip())
            if len(fields) >= 2:
                current_table.append(fields)
            elif current_table and len(current_table) > 1:
                tables.append(current_table)
                current_table = []
        
        if current_table and len(current_table) > 1:
            tables.append(current_table)
        
        return tables
    
    def _get_expected_fields(self, document_type: str) -> List[str]:
        """Get expected fields for document type"""
        field_map = {
            "purchase_order": ["po_number", "po_date", "seller_name", "buyer_name"],
            "invoice": ["invoice_number", "invoice_date", "total_amount"],
            "proforma_invoice": ["invoice_number", "invoice_date", "incoterm"],
            "bank_guarantee": ["guarantee_number", "issue_date", "guarantee_amount"],
            "lc_application": ["lc_number", "application_date", "lc_amount"]
        }
        
        return field_map.get(document_type, [])
    
    def _validate_field(self, field_name: str, value: Any, document_type: str) -> Dict[str, Any]:
        """Validate field format"""
        validation = {"is_valid": True, "message": "Valid", "confidence": 1.0}
        
        if "date" in field_name.lower():
            date_pattern = r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$'
            if not re.match(date_pattern, str(value)):
                validation["is_valid"] = False
                validation["message"] = "Invalid date format"
                validation["confidence"] = 0.3
        
        elif "number" in field_name.lower():
            if not re.match(r'^[A-Z0-9-]+$', str(value), re.IGNORECASE):
                validation["is_valid"] = False
                validation["message"] = "Invalid number format"
                validation["confidence"] = 0.5
        
        elif "amount" in field_name.lower():
            amount_pattern = r'^[$₹€£]?\s*[\d,]+\.?\d*$'
            if not re.match(amount_pattern, str(value)):
                validation["is_valid"] = False
                validation["message"] = "Invalid amount format"
                validation["confidence"] = 0.4
        
        return validation
    
    def _calculate_confidence(self, entities: Dict[str, Any]) -> float:
        """Calculate extraction confidence"""
        if not entities:
            return 0.0
        
        field_count = len(entities)
        non_empty_count = sum(1 for v in entities.values() if v and str(v).strip())
        
        if field_count == 0:
            return 0.0
        
        base_confidence = non_empty_count / field_count
        
        # Bonus for key fields
        key_fields = ["po_number", "invoice_number", "total_amount"]
        key_field_bonus = sum(0.1 for field in key_fields 
                            if field in entities and entities[field])
        
        return min(1.0, base_confidence + key_field_bonus)
    
    def _calculate_completeness(self, entities: Dict[str, Any],
                              expected_fields: List[str]) -> float:
        """Calculate completeness score"""
        if not expected_fields:
            return 1.0
        
        present_fields = sum(1 for field in expected_fields
                           if field in entities and entities[field])
        
        return round(present_fields / len(expected_fields), 2)

async def main():
    """Run the entity extraction MCP server"""
    server = EntityExtractionMCPServer()
    logger.info("🚀 Entity Extraction MCP Server starting...")
    
    # Run server
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
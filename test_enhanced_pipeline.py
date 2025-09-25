#!/usr/bin/env python3
"""
Test the enhanced multi-PDF entity extraction with Document2.pdf
"""
import requests
import json

def test_full_pipeline():
    """Test the complete pipeline with Document2.pdf"""
    
    url = "http://localhost:8000/process"
    file_path = "testing_documents/Document2.pdf"
    
    print("Testing enhanced multi-PDF entity extraction pipeline...")
    print(f"File: {file_path}")
    print("="*60)
    
    with open(file_path, "rb") as f:
        files = {'file': ('Document2.pdf', f, 'application/pdf')}
        data = {
            'enable_classification': 'true',
            'enable_entity_extraction': 'true',
            'enable_preprocessing': 'false',
            'enable_quality_check': 'false',
            'enable_rule_engine': 'false'
        }
        
        try:
            response = requests.post(url, files=files, data=data, timeout=300)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Show classification results
                if "classification" in result.get("results", {}):
                    classification = result["results"]["classification"]["result"]
                    print(f"\n📊 CLASSIFICATION RESULTS:")
                    print(f"Document: {classification.get('document_name', 'Unknown')}")
                    print(f"Total Categories: {classification.get('total_categories', 0)}")
                    print(f"Total PDFs Created: {classification.get('total_pdfs_created', 0)}")
                    
                    if "pdf_summary" in classification:
                        print(f"\n📁 PDF SUMMARY:")
                        for category, info in classification["pdf_summary"].items():
                            print(f"  {category}: {len(info['pdf_files'])} PDFs, {info['page_count']} pages")
                
                # Show entity extraction results
                if "entity_extraction" in result.get("results", {}):
                    entities = result["results"]["entity_extraction"]
                    print(f"\n🔍 ENTITY EXTRACTION RESULTS:")
                    print(f"Session ID: {entities.get('session_id', 'Unknown')}")
                    print(f"Total Categories: {entities.get('total_categories', 0)}")
                    print(f"Total PDFs Processed: {entities.get('total_pdfs_processed', 0)}")
                    
                    if "categories" in entities:
                        print(f"\n📄 PROCESSED CATEGORIES:")
                        for category, cat_info in entities["categories"].items():
                            print(f"\n  📂 {category}:")
                            print(f"    PDFs: {len(cat_info.get('pdf_files', []))}")
                            print(f"    Total Entities: {cat_info.get('total_entities', 0)}")
                            
                            # Show entities for each PDF
                            for pdf_info in cat_info.get("pdf_files", []):
                                print(f"\n    📄 {pdf_info.get('filename', 'Unknown')}:")
                                print(f"      Pages: {len(pdf_info.get('pages', []))}")
                                print(f"      Entities: {len(pdf_info.get('entities', {}))}")
                                print(f"      Confidence: {pdf_info.get('confidence', 0.0):.2f}")
                                print(f"      Processing Time: {pdf_info.get('processing_time', 0.0):.2f}s")
                                
                                # Show some entities
                                entities_data = pdf_info.get('entities', {})
                                if entities_data:
                                    print(f"      🏷️  Entity Sample:")
                                    for i, (entity_type, entity_info) in enumerate(entities_data.items()):
                                        if i >= 3:  # Show only first 3 entities
                                            break
                                        value = entity_info.get('value', 'N/A') if isinstance(entity_info, dict) else entity_info
                                        boxes = entity_info.get('bounding_boxes', []) if isinstance(entity_info, dict) else []
                                        print(f"        {entity_type}: {value} ({len(boxes)} boxes)")
                
                # Show any errors
                if "errors" in result:
                    print(f"\n❌ ERRORS:")
                    for service, error in result["errors"].items():
                        print(f"  {service}: {error}")
                        
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    test_full_pipeline()
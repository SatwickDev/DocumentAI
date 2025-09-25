#!/usr/bin/env python3
"""
Test bounding boxes with different document types
"""
import requests
import json
import os

def test_different_document_types():
    """Test bounding boxes with different document types"""
    
    url = "http://localhost:8000/extract-entities/batch"
    
    # Test with generic document type (not purchase_order)
    files = []
    file_paths = ["test_purchase_order.pdf"]
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            files.append(('files', (os.path.basename(file_path), open(file_path, 'rb'), 'application/pdf')))
            print(f"✓ Found file: {file_path}")
    
    if not files:
        print("No files found!")
        return
    
    # Test with different document types
    test_types = ["invoice", "bank_guarantee", "proforma_invoice"]
    
    for doc_type in test_types:
        print(f"\n{'='*50}")
        print(f"Testing with document type: {doc_type}")
        print('='*50)
        
        data = {'document_type': doc_type}
        
        try:
            # Reset file pointers
            for file_tuple in files:
                file_tuple[1][1].seek(0)
            
            response = requests.post(url, files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                
                for doc_result in result.get("results", []):
                    if doc_result["status"] == "success":
                        entities = doc_result["data"]["entities"]
                        bbox_count = 0
                        
                        for category, entity_list in entities.items():
                            if isinstance(entity_list, list):
                                for entity in entity_list:
                                    if entity.get('bbox'):
                                        bbox_count += 1
                                        print(f"Found bbox for {entity.get('label', 'Unknown')}: {entity['bbox']}")
                        
                        print(f"Total entities with bounding boxes: {bbox_count}")
                        
                        # Show first entity with bbox if any
                        found_bbox = False
                        for category, entity_list in entities.items():
                            if isinstance(entity_list, list):
                                for entity in entity_list:
                                    if entity.get('bbox'):
                                        print(f"Sample entity with bbox:")
                                        print(f"  Label: {entity.get('label', 'N/A')}")
                                        print(f"  Value: {entity.get('value', 'N/A')}")
                                        print(f"  BBox: {entity['bbox']}")
                                        found_bbox = True
                                        break
                            if found_bbox:
                                break
                        
                        if bbox_count == 0:
                            print("No entities have bounding boxes")
                    else:
                        print(f"Extraction failed: {doc_result.get('error', 'Unknown error')}")
                        
            else:
                print(f"Request failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Close file handles
    for file_tuple in files:
        file_tuple[1][1].close()

if __name__ == "__main__":
    test_different_document_types()
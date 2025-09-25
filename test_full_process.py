#!/usr/bin/env python3
"""
Test the full process workflow with classification
"""
import requests
import json
import os

def test_full_process():
    """Test the full /process workflow"""
    
    url = "http://localhost:8000/process"
    
    file_path = "test_purchase_order.pdf"
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return
    
    with open(file_path, "rb") as f:
        files = {'file': (f.name, f, 'application/pdf')}
        data = {
            'enable_quality_analysis': 'true',
            'enable_classification': 'true',
            'enable_entity_extraction': 'true',
            'enable_rule_validation': 'true'
        }
        
        try:
            print("Testing full process workflow...")
            response = requests.post(url, files=files, data=data, timeout=120)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Check classification results
                if 'classification' in result:
                    class_result = result['classification']
                    if 'result' in class_result:
                        category = class_result['result'].get('classification', 'N/A')
                        confidence = class_result['result'].get('confidence_percentage', 'N/A')
                        print(f"✓ Classification: {category} (confidence: {confidence})")
                    else:
                        print("✗ No classification result found")
                else:
                    print("✗ No classification in response")
                
                # Check entity extraction results
                if 'entities' in result:
                    entities = result['entities']
                    entity_count = sum(len(v) if isinstance(v, list) else 0 for v in entities.values())
                    print(f"✓ Entities extracted: {entity_count}")
                    
                    # Check for bounding boxes
                    bbox_count = 0
                    for category, entity_list in entities.items():
                        if isinstance(entity_list, list):
                            for entity in entity_list:
                                if entity.get('bbox'):
                                    bbox_count += 1
                    print(f"✓ Entities with bounding boxes: {bbox_count}")
                else:
                    print("✗ No entities in response")
                
                # Check processing steps
                if 'processing_history' in result:
                    steps = result['processing_history']
                    print(f"✓ Processing steps completed: {len(steps)}")
                    for step in steps:
                        service = step.get('service', 'Unknown')
                        status = step.get('status', 'unknown')
                        print(f"    {service}: {status}")
                
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    test_full_process()
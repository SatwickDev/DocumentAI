#!/usr/bin/env python3
"""
Test the coordinate extraction function directly
"""
import requests
import json

def test_coordinate_extraction():
    """Test if the coordinate extraction is working"""
    
    # Test the entity extraction service directly to see raw response
    url = "http://localhost:8004/extract"
    
    with open("test_purchase_order.pdf", "rb") as f:
        files = {'file': (f.name, f, 'application/pdf')}
        data = {'document_type': 'purchase_order'}
        
        response = requests.post(url, files=files, data=data, timeout=120)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nRaw entity extraction service response:")
            print("=" * 60)
            print(json.dumps(result, indent=2, default=str))
            print("=" * 60)
            
            # Check if there are bounding boxes in the response
            entities = result.get("entities", {})
            bbox_count = 0
            for category, entity_list in entities.items():
                if isinstance(entity_list, list):
                    for entity in entity_list:
                        if entity.get('bbox'):
                            bbox_count += 1
            
            print(f"\nEntities with bounding boxes: {bbox_count}")
            
            # Check if bounding_boxes_included flag is present
            if result.get("bounding_boxes_included"):
                print("✓ Bounding boxes are supposed to be included")
            else:
                print("✗ Bounding boxes flag not set")
                
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    test_coordinate_extraction()
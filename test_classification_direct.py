#!/usr/bin/env python3
"""
Test classification service directly
"""
import requests
import os

def test_classification_service():
    """Test the classification service directly"""
    
    url = "http://localhost:8001/classify"
    
    # Test with the same PDF file
    file_path = "test_purchase_order.pdf"
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return
    
    with open(file_path, "rb") as f:
        files = {'file': (f.name, f, 'application/pdf')}
        
        try:
            print("Testing classification service directly...")
            response = requests.post(url, files=files, timeout=60)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("Classification result:")
                print(f"  Document type: {result.get('document_type', 'N/A')}")
                print(f"  Confidence: {result.get('confidence', 'N/A')}")
                print(f"  Method: {result.get('classification_method', 'N/A')}")
                
                if 'page_results' in result:
                    print(f"  Pages processed: {len(result['page_results'])}")
                    for i, page in enumerate(result['page_results'][:3]):  # First 3 pages
                        print(f"    Page {i+1}: {page.get('prediction', 'N/A')} (confidence: {page.get('confidence', 'N/A')})")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    test_classification_service()
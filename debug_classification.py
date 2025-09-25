#!/usr/bin/env python3
"""
Debug classification service response
"""
import requests
import json
import os

def debug_classification_response():
    """Debug the full classification response"""
    
    url = "http://localhost:8001/classify"
    
    file_path = "test_purchase_order.pdf"
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return
    
    with open(file_path, "rb") as f:
        files = {'file': (f.name, f, 'application/pdf')}
        
        try:
            response = requests.post(url, files=files, timeout=60)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n" + "="*50)
                print("FULL CLASSIFICATION RESPONSE:")
                print("="*50)
                print(json.dumps(result, indent=2, default=str))
                print("="*50)
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    debug_classification_response()
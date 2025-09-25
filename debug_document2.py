#!/usr/bin/env python3
"""
Debug full response for Document2.pdf classification
"""
import requests
import json
import os

def debug_document2_response():
    """See the full response structure for Document2.pdf"""
    
    url = "http://localhost:8001/classify"
    file_path = "testing_documents/Document2.pdf"
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return
    
    with open(file_path, "rb") as f:
        files = {'file': ('Document2.pdf', f, 'application/pdf')}
        
        try:
            response = requests.post(url, files=files, timeout=120)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n" + "="*60)
                print("FULL DOCUMENT2.PDF RESPONSE:")
                print("="*60)
                print(json.dumps(result, indent=2, default=str))
                print("="*60)
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    debug_document2_response()
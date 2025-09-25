#!/usr/bin/env python3
"""
Debug the API response structure
"""
import requests
import json

def debug_response():
    """Check what the API is actually returning"""
    
    file_path = "testing_documents/Document2.pdf"
    url = "http://localhost:8000/process"
    
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
            response = requests.post(url, files=files, data=data, timeout=120)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n" + "="*60)
                print("FULL RESPONSE STRUCTURE:")
                print("="*60)
                print(json.dumps(result, indent=2, default=str))
                print("="*60)
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    debug_response()
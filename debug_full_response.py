#!/usr/bin/env python3
"""
Show the full API response structure
"""
import requests
import json

def show_full_response():
    """Show the complete API response to debug the issue"""
    
    file_path = "testing_documents/Document2.pdf"
    url = "http://localhost:8000/process"
    
    print("🔍 Full API Response Debug...")
    
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
            response = requests.post(url, files=files, data=data, timeout=60)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n" + "="*70)
                print("COMPLETE API RESPONSE:")
                print("="*70)
                print(json.dumps(result, indent=2, default=str))
                print("="*70)
            else:
                print(f"Error Response: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    show_full_response()
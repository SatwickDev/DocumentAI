#!/usr/bin/env python3
"""
Debug script to see the full batch extraction response
"""
import requests
import json
import os

def debug_batch_response():
    """Debug the batch entity extraction response"""
    
    url = "http://localhost:8000/extract-entities/batch"
    
    # Prepare files
    files = []
    file_paths = ["test_purchase_order.pdf"]
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            files.append(('files', (os.path.basename(file_path), open(file_path, 'rb'), 'application/pdf')))
            print(f"✓ Found file: {file_path}")
    
    if not files:
        print("No files found!")
        return
    
    data = {'document_type': 'purchase_order'}
    
    try:
        print("\nMaking request...")
        response = requests.post(url, files=files, data=data, timeout=120)
        
        # Close file handles
        for file_tuple in files:
            file_tuple[1][1].close()
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*50)
            print("FULL RESPONSE:")
            print("="*50)
            print(json.dumps(result, indent=2, default=str))
            print("="*50)
            
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_batch_response()
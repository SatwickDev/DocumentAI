#!/usr/bin/env python3
"""
Test script to directly test entity extraction service batch endpoint
"""
import requests
import os

def test_entity_service_batch():
    """Test the entity extraction service batch endpoint directly"""
    
    # Entity extraction service endpoint
    url = "http://localhost:8004/extract/batch"
    
    # Prepare files
    files = []
    file_paths = [
        "test_purchase_order.pdf",
        "testing_documents/Document2.pdf"
    ]
    
    # Check if files exist and prepare them
    for file_path in file_paths:
        if os.path.exists(file_path):
            files.append(('files', (os.path.basename(file_path), open(file_path, 'rb'), 'application/pdf')))
            print(f"✓ Found file: {file_path}")
        else:
            print(f"✗ File not found: {file_path}")
    
    if not files:
        print("No files found to test!")
        return
    
    # Prepare data
    data = {
        'document_type': 'purchase_order'
    }
    
    try:
        print(f"\nTesting entity extraction service batch endpoint with {len(files)} files...")
        print("Making request to:", url)
        
        # Make the request
        response = requests.post(url, files=files, data=data, timeout=120)
        
        # Close file handles
        for file_tuple in files:
            file_tuple[1][1].close()
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Batch extraction successful!")
            print("Response structure keys:", list(result.keys()))
            
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print("Response:", response.text[:500])
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

if __name__ == "__main__":
    print("Entity Extraction Service Batch Test")
    print("=" * 40)
    test_entity_service_batch()
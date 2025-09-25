#!/usr/bin/env python3
"""
Test script for multi-document entity extraction with bounding boxes
"""
import requests
import json
import os

def test_batch_entity_extraction():
    """Test the batch entity extraction endpoint"""
    
    # API endpoint
    url = "http://localhost:8000/extract-entities/batch"
    print(f"Testing endpoint: {url}")
    
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
        print(f"\nTesting batch entity extraction with {len(files)} files...")
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
            
            # Check if we have results
            if 'results' in result:
                print(f"Number of results: {len(result['results'])}")
                
                # Check each result for bounding boxes
                for i, doc_result in enumerate(result['results']):
                    filename = doc_result.get('filename', f'Document {i+1}')
                    print(f"\nDocument: {filename}")
                    print(f"  Status: {doc_result.get('status', 'unknown')}")
                    print(f"  Confidence: {doc_result.get('confidence', 'N/A')}")
                    
                    # Check entities and bounding boxes
                    entities = doc_result.get('entities', {})
                    bbox_count = 0
                    for category, entity_list in entities.items():
                        if isinstance(entity_list, list):
                            for entity in entity_list:
                                if entity.get('bbox'):
                                    bbox_count += 1
                    
                    print(f"  Total entities: {sum(len(v) if isinstance(v, list) else 0 for v in entities.values())}")
                    print(f"  Entities with bounding boxes: {bbox_count}")
                    
                    # Print sample entities
                    for category, entity_list in list(entities.items())[:2]:  # First 2 categories
                        if isinstance(entity_list, list) and entity_list:
                            print(f"  {category} samples:")
                            for entity in entity_list[:2]:  # First 2 entities
                                bbox_info = " (with bbox)" if entity.get('bbox') else " (no bbox)"
                                print(f"    - {entity.get('label', 'N/A')}: {entity.get('value', 'N/A')}{bbox_info}")
            
            # Check summary
            if 'summary' in result:
                summary = result['summary']
                print(f"\nSummary:")
                print(f"  Total documents: {summary.get('total_documents', 0)}")
                print(f"  Successful: {summary.get('successful_extractions', 0)}")
                print(f"  Failed: {summary.get('failed_extractions', 0)}")
                print(f"  Average confidence: {summary.get('average_confidence', 0):.2f}")
            
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print("Response:", response.text[:500])
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

if __name__ == "__main__":
    print("Multi-Document Entity Extraction Test")
    print("=" * 40)
    test_batch_entity_extraction()
#!/usr/bin/env python3
"""
Debug script to test entity extraction endpoint and response format
"""
import requests
import json
import os

def test_api_gateway_response():
    """Test what the API Gateway returns for entity extraction"""
    print("Testing API Gateway entity extraction response...")
    
    # Test with test_purchase_order.pdf - this should trigger entity extraction
    files = {"file": ("test_purchase_order.pdf", open("test_purchase_order.pdf", "rb"), "application/pdf")}
    data = {
        "enable_classification": True,
        "enable_entity_extraction": True,
        "enable_preprocessing": True,
        "enable_quality": True
    }
    
    try:
        response = requests.post("http://localhost:8000/process", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response status: {response.status_code}")
            print(f"Response keys: {list(result.keys())}")
            
            if "results" in result:
                print(f"Results keys: {list(result['results'].keys())}")
                
                if "entity_extraction" in result["results"]:
                    entity_data = result["results"]["entity_extraction"]
                    print(f"Entity extraction keys: {list(entity_data.keys())}")
                    print(f"Entity extraction response structure:")
                    print(json.dumps(entity_data, indent=2)[:1000] + "..." if len(json.dumps(entity_data)) > 1000 else json.dumps(entity_data, indent=2))
                    
                    # Check if it has categories
                    if "categories" in entity_data:
                        print(f"✅ Categories found: {list(entity_data['categories'].keys())}")
                    else:
                        print("❌ No 'categories' key found in entity extraction response")
                else:
                    print("❌ No 'entity_extraction' key found in results")
            else:
                print("❌ No 'results' key found in response")
                
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing API Gateway: {str(e)}")
    finally:
        files["file"][1].close()  # Close file handle

def check_directory_structure():
    """Check if the document_classification_updated directory exists"""
    print("\nChecking directory structure...")
    
    base_dir = "document_classification_updated"
    if os.path.exists(base_dir):
        print(f"✅ {base_dir} exists")
        
        for doc_folder in os.listdir(base_dir):
            doc_path = os.path.join(base_dir, doc_folder)
            if os.path.isdir(doc_path):
                print(f"  📁 Document folder: {doc_folder}")
                
                for category_folder in os.listdir(doc_path):
                    category_path = os.path.join(doc_path, category_folder)
                    if os.path.isdir(category_path):
                        pdf_files = [f for f in os.listdir(category_path) if f.lower().endswith('.pdf')]
                        print(f"    📂 Category: {category_folder} ({len(pdf_files)} PDFs)")
                        for pdf in pdf_files[:3]:  # Show first 3 PDFs
                            print(f"      📄 {pdf}")
                        if len(pdf_files) > 3:
                            print(f"      ... and {len(pdf_files) - 3} more PDFs")
    else:
        print(f"❌ {base_dir} does not exist")

if __name__ == "__main__":
    check_directory_structure()
    test_api_gateway_response()
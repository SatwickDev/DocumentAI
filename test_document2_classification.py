#!/usr/bin/env python3
"""
Test classification with Document2.pdf from testing_documents folder
"""
import requests
import json
import os

def test_document2_classification():
    """Test classification service with Document2.pdf"""
    
    url = "http://localhost:8001/classify"
    
    # Test with Document2.pdf from testing_documents
    file_path = "testing_documents/Document2.pdf"
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return
    
    print(f"Testing classification with: {file_path}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    with open(file_path, "rb") as f:
        files = {'file': ('Document2.pdf', f, 'application/pdf')}
        
        try:
            print("Sending classification request...")
            response = requests.post(url, files=files, timeout=120)
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n" + "="*60)
                print("DOCUMENT2.PDF CLASSIFICATION RESULT")
                print("="*60)
                
                # Extract key information
                if 'result' in result:
                    res = result['result']
                    print(f"Document Type: {res.get('classification', 'N/A')}")
                    print(f"Category: {res.get('category', 'N/A')}")
                    print(f"Confidence: {res.get('confidence_percentage', 'N/A')}")
                    print(f"Method: {res.get('method', 'N/A')}")
                    print(f"Technique Used: {res.get('technique_used', 'N/A')}")
                    print(f"Pages Processed: {res.get('pages_processed', 'N/A')}")
                    print(f"Document Name: {res.get('document_name', 'N/A')}")
                    
                    # Show output directory
                    output_dir = res.get('output_directory', 'N/A')
                    print(f"Output Directory: {output_dir}")
                    
                    # Show page-by-page results
                    if 'detailed_results' in res:
                        print(f"\nPage-by-page Results:")
                        for page_result in res['detailed_results']:
                            page_num = page_result.get('page_num', 'N/A')
                            category = page_result.get('category', 'N/A')
                            confidence = page_result.get('confidence', 'N/A')
                            technique = page_result.get('technique', 'N/A')
                            print(f"  Page {page_num}: {category} ({confidence}) - {technique}")
                    
                    # Show created PDFs
                    if 'created_pdfs' in res:
                        print(f"\nCreated PDF Files:")
                        for pdf_info in res['created_pdfs']:
                            page_num = pdf_info.get('Page Number', 'N/A')
                            category = pdf_info.get('Final Category', 'N/A')
                            confidence = pdf_info.get('Confidence Score', 'N/A')
                            filename = pdf_info.get('Output File', 'N/A')
                            print(f"  Page {page_num}: {filename} -> {category} ({confidence})")
                
                print("="*60)
                
            else:
                print(f"Classification failed with status {response.status_code}")
                print(f"Error response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("Request timed out - classification may take longer for this document")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    test_document2_classification()
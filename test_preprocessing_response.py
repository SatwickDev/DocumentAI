#!/usr/bin/env python3
"""
Test script to check preprocessing data in quality service response
"""
import requests
import json
import os

def test_quality_response():
    # Test with available PDF files
    test_files = [
        "test_purchase_order.pdf",  # Available PDF
        "test_multi_page.pdf",      # Multi-page PDF for testing
        "sample.pdf"                # Any sample PDF
    ]
    
    quality_url = "http://localhost:8002/analyze"
    
    print("🔍 Testing Quality Service Response for Preprocessing Data...")
    
    # First check if service is accessible
    try:
        health_response = requests.get("http://localhost:8002/health")
        print(f"✅ Quality Service Health: {health_response.status_code}")
    except Exception as e:
        print(f"❌ Cannot reach quality service: {e}")
        return
    
    # Find a test file
    test_file = None
    for filename in test_files:
        if os.path.exists(filename):
            test_file = filename
            break
    
    if not test_file:
        print("⚠️  No test PDF files found. Please ensure you have a test PDF file.")
        print("   Expected files: test_multi_page.pdf or sample.pdf")
        return
    
    print(f"📄 Using test file: {test_file}")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'application/pdf')}
            
            print("📤 Sending request to quality service...")
            response = requests.post(quality_url, files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Quality analysis successful!")
                
                # Print full response structure for debugging
                print(f"\n🔍 Full response structure:")
                for key in result.keys():
                    if key == 'result' and isinstance(result[key], dict):
                        print(f"  {key}: {list(result[key].keys())}")
                    else:
                        print(f"  {key}: {type(result[key])}")
                
                # Check for page_results in different locations
                page_results = None
                if 'page_results' in result:
                    page_results = result['page_results']
                elif 'result' in result and isinstance(result['result'], dict):
                    if 'page_results' in result['result']:
                        page_results = result['result']['page_results']
                
                if page_results:
                    print(f"\n📊 Found {len(page_results)} pages")
                    
                    for i, page in enumerate(page_results):
                        print(f"\n--- Page {page.get('page', i+1)} ---")
                        print(f"Score: {page.get('score', 'N/A')}")
                        print(f"Verdict: {page.get('verdict', 'N/A')}")
                        
                        # Print all page keys for debugging
                        print(f"Page keys: {list(page.keys())}")
                        
                        # Check for preprocessing data
                        if 'preprocessing' in page:
                            print("✅ Preprocessing data found:")
                            preprocessing = page['preprocessing']
                            print(f"  Status: {preprocessing.get('status', 'N/A')}")
                            print(f"  Message: {preprocessing.get('message', 'N/A')}")
                            print(f"  Color: {preprocessing.get('color', 'N/A')}")
                            print(f"  Icon: {preprocessing.get('icon', 'N/A')}")
                        else:
                            print("❌ No preprocessing data found in page")
                
                else:
                    print("❌ No page_results found anywhere in response")
                
            else:
                print(f"❌ Quality analysis failed: {response.status_code}")
                print(f"Response: {response.text}")
                
    except FileNotFoundError:
        print(f"❌ Test file not found: {test_file}")
    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    test_quality_response()
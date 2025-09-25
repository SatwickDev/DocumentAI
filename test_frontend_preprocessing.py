#!/usr/bin/env python3
"""
Frontend preprocessing test - verify the updated preprocessing interface
"""
import requests
import json

def test_frontend_preprocessing():
    print("🧪 Testing Frontend Preprocessing Interface")
    print("=" * 50)
    
    # Test the API Gateway which the frontend uses
    api_url = "http://localhost:8000/process"
    
    try:
        with open("test_purchase_order.pdf", 'rb') as f:
            files = {'file': ('test_purchase_order.pdf', f, 'application/pdf')}
            data = {
                'session_id': 'test-frontend-preprocessing',
                'enable_quality': 'true',
                'enable_entity_extraction': 'false',
                'enable_classification': 'false'
            }
            
            print("📤 Sending request to API Gateway (like frontend does)...")
            response = requests.post(api_url, files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API Gateway response successful!")
                
                # Check quality results structure
                if 'quality' in result and 'result' in result['quality']:
                    quality_result = result['quality']['result']
                    
                    if 'page_results' in quality_result:
                        pages = quality_result['page_results']
                        print(f"\n📊 Quality Analysis: {len(pages)} page(s)")
                        
                        for i, page in enumerate(pages):
                            print(f"\n--- Page {page.get('page', i+1)} ---")
                            print(f"  Score: {(page.get('score', 0) * 100):.1f}%")
                            print(f"  Verdict: {page.get('verdict', 'N/A')}")
                            
                            if 'preprocessing' in page:
                                prep = page['preprocessing']
                                print("  ✅ Preprocessing Data Found:")
                                print(f"    Status: {prep.get('status', 'N/A')}")
                                print(f"    Reason: {prep.get('reason', 'N/A')}")
                                print(f"    Recommended: {prep.get('recommended', 'N/A')}")
                                print(f"    Operations: {prep.get('operations', [])}")
                                print(f"    Icon: {prep.get('status_icon', 'N/A')}")
                                print(f"    Color: {prep.get('status_color', 'N/A')}")
                                print(f"    Time: {prep.get('estimated_time', 0)}s")
                            else:
                                print("  ❌ No preprocessing data found")
                    else:
                        print("❌ No page_results found in quality response")
                else:
                    print("❌ No quality results found in response")
                    
            else:
                print(f"❌ API Gateway request failed: {response.status_code}")
                print(f"Response: {response.text}")
                
    except FileNotFoundError:
        print("❌ Test file not found: test_purchase_order.pdf")
    except Exception as e:
        print(f"❌ Error during test: {e}")

    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Open http://localhost:8080 in your browser")
    print("2. Upload the test_purchase_order.pdf file")
    print("3. Run quality analysis")
    print("4. Check the 'Per-Page Processing Assessment' section")
    print("5. Click on page buttons to see preprocessing details")

if __name__ == "__main__":
    test_frontend_preprocessing()
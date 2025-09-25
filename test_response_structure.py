#!/usr/bin/env python3
"""
Quick test to capture the actual API response structure
"""
import requests
import json

def test_api_response():
    """Test the actual API response structure"""
    print("Testing API response structure...")
    
    try:
        # Test with test_purchase_order.pdf
        with open("test_purchase_order.pdf", "rb") as f:
            files = {"file": ("test_purchase_order.pdf", f, "application/pdf")}
            data = {
                "enable_classification": True,
                "enable_entity_extraction": True,
                "enable_preprocessing": True,
                "enable_quality": True
            }
            
            response = requests.post("http://localhost:8000/process", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response status: {response.status_code}")
                print(f"📋 Top-level keys: {list(result.keys())}")
                
                if "results" in result:
                    results = result["results"]
                    print(f"📋 Results keys: {list(results.keys())}")
                    
                    if "entity_extraction" in results:
                        entity_data = results["entity_extraction"]
                        print(f"📋 Entity extraction keys: {list(entity_data.keys())}")
                        
                        # Check what the frontend is looking for
                        if "categories" in entity_data:
                            print(f"✅ FOUND 'categories' key!")
                            print(f"📂 Categories: {list(entity_data['categories'].keys())}")
                            
                            # Show the structure frontend expects
                            print(f"🔍 Frontend expects: lastResult.results.entity_extraction.categories")
                            print(f"🔍 We have: result.results.entity_extraction.categories ✅")
                            
                            # Show some sample data
                            for cat_name, cat_data in entity_data['categories'].items():
                                print(f"   📂 {cat_name}: {len(cat_data.get('pdf_files', []))} PDFs")
                        else:
                            print(f"❌ No 'categories' key found")
                            print(f"🔍 Available keys: {list(entity_data.keys())}")
                    else:
                        print(f"❌ No 'entity_extraction' key in results")
                        print(f"🔍 Available result keys: {list(results.keys())}")
                else:
                    print(f"❌ No 'results' key in response")
                    print(f"🔍 Available keys: {list(result.keys())}")
                    
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"Error: {response.text}")
                
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_api_response()
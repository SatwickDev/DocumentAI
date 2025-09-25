#!/usr/bin/env python3
"""
Debug API response structure to see why frontend isn't showing entity extraction
"""
import requests
import json

def debug_api_response():
    """Check the exact structure of the API response"""
    
    file_path = "testing_documents/Document2.pdf"
    url = "http://localhost:8000/process"
    
    print("🔍 Debugging API Response Structure...")
    
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
            print("Making API request...")
            response = requests.post(url, files=files, data=data, timeout=90)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                print("\n" + "="*70)
                print("📊 RESPONSE STRUCTURE ANALYSIS")
                print("="*70)
                
                # Show top-level keys
                print(f"Top-level keys: {list(result.keys())}")
                
                # Check if 'results' exists
                if 'results' in result:
                    print(f"results keys: {list(result['results'].keys())}")
                    
                    # Check entity_extraction specifically
                    if 'entity_extraction' in result['results']:
                        entity_data = result['results']['entity_extraction']
                        print(f"entity_extraction keys: {list(entity_data.keys())}")
                        
                        # Show a condensed version of entity data
                        print(f"\nEntity extraction summary:")
                        print(f"- total_pdfs_processed: {entity_data.get('total_pdfs_processed', 'MISSING')}")
                        print(f"- total_categories: {entity_data.get('total_categories', 'MISSING')}")
                        print(f"- categories: {list(entity_data.get('categories', {}).keys())}")
                        
                        return True
                    else:
                        print("❌ 'entity_extraction' not found in results")
                else:
                    print("❌ 'results' not found in response")
                
                # Show the full response (truncated for readability)
                print(f"\n" + "="*70)
                print("📄 FULL RESPONSE (first 2000 chars):")
                print("="*70)
                response_str = json.dumps(result, indent=2, default=str)
                print(response_str[:2000])
                if len(response_str) > 2000:
                    print("... (truncated)")
                    
                return False
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return False

if __name__ == "__main__":
    success = debug_api_response()
    print(f"\n🎯 Entity extraction found: {'✅ YES' if success else '❌ NO'}")
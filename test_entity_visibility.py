#!/usr/bin/env python3
"""
Quick test to verify entity extraction is now visible in API response
"""
import requests
import json

def test_entity_visibility():
    """Test if entity extraction results are now visible"""
    
    file_path = "testing_documents/Document2.pdf"
    url = "http://localhost:8000/process"
    
    print("🔍 Testing Entity Extraction Visibility...")
    
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
            response = requests.post(url, files=files, data=data, timeout=120)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Check if entity_extraction is in results
                if "results" in result and "entity_extraction" in result["results"]:
                    entity_data = result["results"]["entity_extraction"]
                    print("✅ Entity extraction found in response!")
                    
                    # Check for multi-PDF format
                    if "categories" in entity_data:
                        print(f"📊 Multi-PDF format detected")
                        print(f"📄 Total PDFs processed: {entity_data.get('total_pdfs_processed', 0)}")
                        print(f"📁 Categories: {list(entity_data.get('categories', {}).keys())}")
                        
                        # Show some details
                        categories = entity_data.get("categories", {})
                        for category, info in categories.items():
                            pdf_count = len(info.get("pdf_files", []))
                            entity_count = info.get("total_entities", 0)
                            print(f"  📂 {category}: {pdf_count} PDFs, {entity_count} entities")
                        
                        return True
                    else:
                        print("📊 Legacy format detected")
                        return True
                else:
                    print("❌ Entity extraction not found in response")
                    print(f"Available keys: {list(result.get('results', {}).keys())}")
                    return False
            else:
                print(f"❌ API Error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return False

if __name__ == "__main__":
    success = test_entity_visibility()
    print(f"\n🎯 Result: {'✅ FIXED' if success else '❌ STILL BROKEN'}")
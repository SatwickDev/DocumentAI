#!/usr/bin/env python3
"""
Quick test of the enhanced multi-PDF entity extraction
"""
import requests
import json

def test_document2():
    """Quick test of Document2.pdf with the enhanced system"""
    
    url = "http://localhost:8000/process"
    file_path = "testing_documents/Document2.pdf"
    
    print("Testing Document2.pdf with enhanced multi-PDF entity extraction...")
    
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
                
                # Show entity extraction summary
                if "entity_extraction" in result.get("results", {}):
                    entities = result["results"]["entity_extraction"]
                    print(f"✅ Entity Extraction Success!")
                    print(f"Total Categories: {entities.get('total_categories', 0)}")
                    print(f"Total PDFs Processed: {entities.get('total_pdfs_processed', 0)}")
                    
                    # Show categories processed
                    categories = entities.get("categories", {})
                    print(f"Categories: {list(categories.keys())}")
                    
                    return True
                else:
                    print("❌ No entity extraction results found")
                    return False
                    
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return False

if __name__ == "__main__":
    success = test_document2()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
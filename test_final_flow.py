#!/usr/bin/env python3
"""
Test the complete enhanced multi-PDF entity extraction flow
"""
import requests
import json
import os

def test_enhanced_flow():
    """Test Document2.pdf with enhanced classification and entity extraction"""
    
    print("=" * 70)
    print("🚀 TESTING ENHANCED MULTI-PDF ENTITY EXTRACTION")
    print("=" * 70)
    
    # Test file
    file_path = "testing_documents/Document2.pdf"
    if not os.path.exists(file_path):
        print(f"❌ Test file not found: {file_path}")
        return False
    
    print(f"📄 Testing file: {file_path}")
    
    # API endpoint
    url = "http://localhost:8000/process"
    
    # Prepare request
    with open(file_path, "rb") as f:
        files = {'file': ('Document2.pdf', f, 'application/pdf')}
        data = {
            'enable_classification': 'true',
            'enable_entity_extraction': 'true',
            'enable_preprocessing': 'false',
            'enable_quality_check': 'false',
            'enable_rule_engine': 'false'
        }
        
        print("\n📡 Sending request to API Gateway...")
        
        try:
            response = requests.post(url, files=files, data=data, timeout=180)
            
            print(f"📊 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Test Classification Results
                print("\n" + "=" * 50)
                print("📋 CLASSIFICATION RESULTS")
                print("=" * 50)
                
                if "classification" in result.get("results", {}):
                    classification = result["results"]["classification"]["result"]
                    print(f"✅ Classification Success!")
                    print(f"📁 Document: {classification.get('document_name', 'Unknown')}")
                    print(f"📊 Categories: {classification.get('total_categories', 0)}")
                    print(f"📄 PDFs Created: {classification.get('total_pdfs_created', 0)}")
                    
                    # Show PDF summary
                    pdf_summary = classification.get("pdf_summary", {})
                    if pdf_summary:
                        print(f"\n📂 FOLDER STRUCTURE:")
                        for category, info in pdf_summary.items():
                            pdf_count = len(info.get('pdf_files', []))
                            page_count = info.get('page_count', 0)
                            print(f"  📁 {category}/ → {pdf_count} PDFs, {page_count} pages")
                else:
                    print("❌ No classification results found")
                    return False
                
                # Test Entity Extraction Results
                print("\n" + "=" * 50)
                print("🔍 ENTITY EXTRACTION RESULTS")
                print("=" * 50)
                
                if "entity_extraction" in result.get("results", {}):
                    entities = result["results"]["entity_extraction"]
                    print(f"✅ Entity Extraction Success!")
                    print(f"📊 Categories Processed: {entities.get('total_categories', 0)}")
                    print(f"📄 PDFs Processed: {entities.get('total_pdfs_processed', 0)}")
                    
                    # Show processing errors if any
                    errors = entities.get("processing_errors", [])
                    if errors:
                        print(f"⚠️  Processing Errors: {len(errors)}")
                        for error in errors:
                            print(f"    ❌ {error}")
                    
                    # Show detailed results by category
                    categories = entities.get("categories", {})
                    if categories:
                        print(f"\n📂 CATEGORY RESULTS:")
                        for category, cat_info in categories.items():
                            print(f"\n  📁 {category}:")
                            pdf_files = cat_info.get("pdf_files", [])
                            print(f"    📄 PDFs: {len(pdf_files)}")
                            print(f"    🏷️  Total Entities: {cat_info.get('total_entities', 0)}")
                            print(f"    ⏱️  Processing Time: {cat_info.get('processing_time', 0.0):.2f}s")
                            
                            # Show details for each PDF
                            for i, pdf_info in enumerate(pdf_files):
                                filename = pdf_info.get("filename", "Unknown")
                                entity_count = len(pdf_info.get("entities", {}))
                                confidence = pdf_info.get("confidence", 0.0)
                                word_boxes = len(pdf_info.get("word_boxes", []))
                                
                                print(f"      📄 {i+1}. {filename}")
                                print(f"         🏷️  Entities: {entity_count}")
                                print(f"         📊 Confidence: {confidence:.2f}")
                                print(f"         📦 Bounding Boxes: {word_boxes}")
                                
                                # Show sample entities
                                entities_data = pdf_info.get("entities", {})
                                if entities_data:
                                    print(f"         🔍 Sample Entities:")
                                    for j, (entity_type, entity_info) in enumerate(entities_data.items()):
                                        if j >= 3:  # Show only first 3
                                            break
                                        if isinstance(entity_info, dict):
                                            value = entity_info.get('value', 'N/A')
                                            boxes = len(entity_info.get('bounding_boxes', []))
                                            print(f"           • {entity_type}: {value} ({boxes} boxes)")
                                        else:
                                            print(f"           • {entity_type}: {entity_info}")
                    else:
                        print("❌ No category results found")
                        return False
                        
                else:
                    print("❌ No entity extraction results found")
                    return False
                
                # Final verification
                total_pdfs = entities.get('total_pdfs_processed', 0)
                expected_pdfs = classification.get('total_pdfs_created', 0)
                
                print(f"\n" + "=" * 50)
                print("✅ VERIFICATION")
                print("=" * 50)
                print(f"Expected PDFs: {expected_pdfs}")
                print(f"Processed PDFs: {total_pdfs}")
                print(f"Match: {'✅ YES' if total_pdfs == expected_pdfs else '❌ NO'}")
                
                return total_pdfs == expected_pdfs and total_pdfs > 0
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return False

if __name__ == "__main__":
    success = test_enhanced_flow()
    print(f"\n" + "=" * 70)
    print(f"🎯 FINAL RESULT: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print("=" * 70)
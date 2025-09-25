#!/usr/bin/env python3
"""Test API with actual PDF processing to debug frontend issues"""

import requests
import json
import os

def test_process_endpoint():
    """Test the /process endpoint with a simple PDF"""
    
    # Create a minimal PDF content for testing
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
trailer
<< /Size 5 /Root 1 0 R >>
startxref
297
%%EOF"""
    
    try:
        print("🔍 Testing /process endpoint with PDF...")
        
        # Prepare the request exactly like the frontend does
        files = {'file': ('test.pdf', pdf_content, 'application/pdf')}
        data = {
            'session_id': 'debug-test-pdf',
            'enable_preprocessing': 'false',
            'enable_entity_extraction': 'false',
            'enable_enhanced_quality': 'true'
        }
        
        response = requests.post('http://localhost:8000/process', files=files, data=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n📋 FULL API RESPONSE:")
            print(json.dumps(result, indent=2))
            
            print(f"\n🔍 FRONTEND DEBUGGING:")
            
            # Test the exact paths the frontend is trying to access
            print(f"- result['quality'] exists: {'quality' in result}")
            
            if 'quality' in result:
                quality = result['quality']
                print(f"- result['quality']['result'] exists: {'result' in quality}")
                
                if 'result' in quality:
                    quality_result = quality['result']
                    print(f"- result['quality']['result']['quality_score'] exists: {'quality_score' in quality_result}")
                    print(f"- result['quality']['result']['verdict'] exists: {'verdict' in quality_result}")
                    
                    if 'quality_score' in quality_result:
                        score = quality_result['quality_score']
                        print(f"- Quality Score Value: {score} (type: {type(score)})")
                        print(f"- Score * 100: {score * 100 if score is not None else 'None'}")
                        print(f"- Score > 0: {score > 0 if score is not None else 'False'}")
                        
                    if 'verdict' in quality_result:
                        verdict = quality_result['verdict']
                        print(f"- Verdict Value: '{verdict}' (type: {type(verdict)})")
                        
            print(f"\n🎯 ANGULAR EXPRESSIONS:")
            if 'quality' in result and 'result' in result['quality']:
                qr = result['quality']['result']
                if 'quality_score' in qr and qr['quality_score'] is not None:
                    score = qr['quality_score']
                    print(f"- {{{{(lastResult.quality.result.quality_score * 100).toFixed(1)}}}}% = {(score * 100):.1f}%")
                else:
                    print(f"- Quality score is None or missing - this causes NaN%")
                    
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_process_endpoint()
#!/usr/bin/env python3
"""
Test script to debug quality analysis scores
"""

import requests
import json

def test_quality_analysis(file_path):
    """Test quality analysis on a specific file"""
    url = "http://localhost:8000/document-processing"
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'analysis_type': 'universal'}
        
        print(f"Testing file: {file_path}")
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract quality results
            quality_result = result.get('quality', {}).get('result', {})
            page_results = quality_result.get('page_results', [])
            
            print(f"Overall Score: {quality_result.get('quality_score', 'N/A')}")
            print(f"Overall Verdict: {quality_result.get('verdict', 'N/A')}")
            print(f"Pages Analyzed: {len(page_results)}")
            print("\nPer-page Results:")
            
            for i, page in enumerate(page_results):
                print(f"  Page {page.get('page', i+1)}: Score={page.get('score', 'N/A'):.3f} ({page.get('score', 0)*100:.1f}%), Verdict={page.get('verdict', 'N/A')}")
                print(f"    Metrics: {json.dumps(page.get('metrics', {}), indent=6)}")
                
        else:
            print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    # Test with the PDF from the root directory
    test_file = "test_purchase_order.pdf"
    test_quality_analysis(test_file)
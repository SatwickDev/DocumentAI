#!/usr/bin/env python3
"""
Debug script to test Universal Analyzer metric extraction
"""

import sys
import os
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "quality_analysis_updated"))

def test_direct_analyzer(file_path):
    """Test Universal Analyzer directly to see raw metrics"""
    try:
        from universal_analyzer import analyze_pdf_fast_parallel, get_metric_value
        
        print(f"Testing file: {file_path}")
        
        # Analyze the document
        analysis_results = analyze_pdf_fast_parallel(file_path, max_workers=2)
        
        print(f"Number of pages analyzed: {len(analysis_results)}")
        
        for i, page_analysis in enumerate(analysis_results):
            print(f"\n=== PAGE {page_analysis.page_num} ===")
            print(f"Has error: {page_analysis.error}")
            print(f"Has metrics: {page_analysis.metrics is not None}")
            
            if page_analysis.metrics:
                metrics = page_analysis.metrics
                print(f"Raw metrics object: {metrics}")
                
                # Test individual metric extraction
                print(f"blur_score raw: {metrics.blur_score}")
                print(f"blur_score extracted: {get_metric_value(metrics.blur_score)}")
                
                print(f"sharpness_score raw: {metrics.sharpness_score}")
                print(f"sharpness_score extracted: {get_metric_value(metrics.sharpness_score)}")
                
                print(f"contrast_score raw: {metrics.contrast_score}")
                print(f"contrast_score extracted: {get_metric_value(metrics.contrast_score)}")
                
                # Extract all metrics the same way as microservice
                metric_dict = {
                    "blur_score": get_metric_value(metrics.blur_score),
                    "contrast_score": get_metric_value(metrics.contrast_score),
                    "noise_level": get_metric_value(metrics.noise_level),
                    "sharpness_score": get_metric_value(metrics.sharpness_score),
                    "brightness_score": get_metric_value(metrics.brightness_score),
                    "skew_angle": get_metric_value(metrics.skew_angle),
                    "edge_crop_score": get_metric_value(metrics.edge_crop_score),
                    "shadow_glare_score": get_metric_value(metrics.shadow_glare_score),
                    "blank_page_score": get_metric_value(metrics.blank_page_score),
                }
                
                print(f"Final metric_dict: {metric_dict}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_file = "quality_analysis_updated/LCSample1.pdf"  # Same file you tested
    test_direct_analyzer(test_file)
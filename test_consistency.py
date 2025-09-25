#!/usr/bin/env python3
"""
Test script to check if quality metrics are consistent for the same input
"""

import sys
import os
sys.path.append('quality_analysis_updated')

from parallel_metrics import analyze_page_metrics_parallel
import numpy as np

def test_metrics_consistency():
    # Create a simple test image
    test_img = np.ones((100, 100), dtype=np.uint8) * 128
    print('Testing metrics consistency...')

    # Run the same analysis multiple times
    results = []
    for i in range(5):
        try:
            metrics = analyze_page_metrics_parallel(test_img)
            blur_score = getattr(metrics.blur_score, 'value', metrics.blur_score) if metrics.blur_score else 0
            contrast_score = getattr(metrics.contrast_score, 'value', metrics.contrast_score) if metrics.contrast_score else 0
            results.append((blur_score, contrast_score))
            print(f'Run {i+1}: blur={blur_score:.6f}, contrast={contrast_score:.6f}')
        except Exception as e:
            print(f'Error in run {i+1}: {e}')

    # Check if all results are identical
    if results:
        all_same = all(r == results[0] for r in results)
        print(f'All results identical: {all_same}')
        if not all_same:
            print('FOUND INCONSISTENCY - this explains the changing quality scores!')
            # Show differences
            for i, result in enumerate(results):
                print(f'  Result {i+1}: {result}')
        else:
            print('Results are consistent - inconsistency might be elsewhere')
    else:
        print('No results collected')

if __name__ == "__main__":
    test_metrics_consistency()
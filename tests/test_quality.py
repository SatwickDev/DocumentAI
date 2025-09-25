#!/usr/bin/env python3
"""
Test Quality Analysis functionality
"""
import unittest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Updated to use Universal Analyzer directly
import sys
sys.path.append('../quality_analysis_updated')
from universal_analyzer import analyze_pdf_fast_parallel
from src.utils.mcp_contracts import QualityRequest, QualityResponse

class TestQualityAnalysis(unittest.TestCase):
    """Test quality analysis functionality"""
    
    def test_quality_basic(self):
        """Test basic quality analysis works"""
        # This is a placeholder for actual tests
        self.assertTrue(True)
        
    def test_quality_contract(self):
        """Test quality contracts work properly"""
        # Create a request
        request = QualityRequest(
            file_path="test.pdf",
            session_id="test-123"
        )
        
        # Verify contract works
        self.assertEqual(request.file_path, "test.pdf")
        self.assertEqual(request.session_id, "test-123")

if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
API endpoint test script
Run this after starting the services to test the endpoints
"""

import requests
import json
import time
import os
from typing import Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
        
    def create_test_file(self) -> bytes:
        """Create a simple test file"""
        # Create a simple text file with purchase order content
        content = """
        PURCHASE ORDER
        
        PO Number: PO-2024-001
        Date: 2024-01-15
        
        Buyer: ABC Company
        123 Main Street
        City, State 12345
        
        Seller: XYZ Supplies
        456 Business Ave
        City, State 67890
        
        Items:
        1. Product A - Quantity: 10 - Price: $50.00
        2. Product B - Quantity: 5 - Price: $100.00
        
        Total Amount: $1,000.00
        
        Delivery Terms: FOB Destination
        Payment Terms: Net 30 days
        """
        return content.encode()
    
    def test_health_check(self):
        """Test health check endpoint"""
        print("\n🔍 Testing Health Check...")
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health Check: {data['status']}")
                print(f"   Services: {json.dumps(data['services'], indent=2)}")
                self.test_results.append(("Health Check", "PASSED"))
            else:
                print(f"❌ Health Check Failed: {response.status_code}")
                self.test_results.append(("Health Check", f"FAILED: {response.status_code}"))
        except Exception as e:
            print(f"❌ Health Check Error: {e}")
            self.test_results.append(("Health Check", f"ERROR: {e}"))
    
    def test_quality_analysis(self):
        """Test quality analysis endpoint"""
        print("\n🔍 Testing Quality Analysis...")
        try:
            files = {"file": ("test.txt", self.create_test_file(), "text/plain")}
            params = {"apply_preprocessing": False}
            
            response = requests.post(
                f"{self.base_url}/analyze/quality",
                files=files,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Quality Analysis Completed")
                print(f"   Overall Score: {data.get('overall_score', 'N/A')}")
                print(f"   Verdict: {data.get('verdict', 'N/A')}")
                self.test_results.append(("Quality Analysis", "PASSED"))
            else:
                print(f"❌ Quality Analysis Failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results.append(("Quality Analysis", f"FAILED: {response.status_code}"))
        except Exception as e:
            print(f"❌ Quality Analysis Error: {e}")
            self.test_results.append(("Quality Analysis", f"ERROR: {e}"))
    
    def test_preprocessing(self):
        """Test preprocessing endpoint"""
        print("\n🔍 Testing Preprocessing...")
        try:
            files = {"file": ("test.txt", self.create_test_file(), "text/plain")}
            params = {
                "operations": "all",
                "return_file": False  # Get base64 response
            }
            
            response = requests.post(
                f"{self.base_url}/preprocess",
                files=files,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Preprocessing Completed")
                print(f"   Status: {data.get('status', 'N/A')}")
                print(f"   Operations: {data.get('operations_applied', [])}")
                self.test_results.append(("Preprocessing", "PASSED"))
            else:
                print(f"❌ Preprocessing Failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results.append(("Preprocessing", f"FAILED: {response.status_code}"))
        except Exception as e:
            print(f"❌ Preprocessing Error: {e}")
            self.test_results.append(("Preprocessing", f"ERROR: {e}"))
    
    def test_entity_extraction(self):
        """Test entity extraction endpoint"""
        print("\n🔍 Testing Entity Extraction...")
        try:
            files = {"file": ("test.txt", self.create_test_file(), "text/plain")}
            params = {
                "document_type": "purchase_order",
                "extract_tables": True
            }
            
            response = requests.post(
                f"{self.base_url}/extract/entities",
                files=files,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Entity Extraction Completed")
                print(f"   Document Type: {data.get('document_type', 'N/A')}")
                print(f"   Confidence: {data.get('confidence', 'N/A')}")
                if 'entities' in data:
                    print(f"   Extracted Entities: {len(data['entities'])} fields")
                self.test_results.append(("Entity Extraction", "PASSED"))
            else:
                print(f"❌ Entity Extraction Failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results.append(("Entity Extraction", f"FAILED: {response.status_code}"))
        except Exception as e:
            print(f"❌ Entity Extraction Error: {e}")
            self.test_results.append(("Entity Extraction", f"ERROR: {e}"))
    
    def test_full_pipeline(self):
        """Test full processing pipeline"""
        print("\n🔍 Testing Full Pipeline...")
        try:
            files = {"file": ("test.txt", self.create_test_file(), "text/plain")}
            
            response = requests.post(
                f"{self.base_url}/process/full-pipeline",
                files=files
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Full Pipeline Completed")
                print(f"   Success Rate: {data.get('success_rate', 0) * 100:.0f}%")
                if 'pipeline_stages' in data:
                    for stage, result in data['pipeline_stages'].items():
                        status = "✓" if isinstance(result, dict) and 'error' not in result else "✗"
                        print(f"   {status} {stage}")
                self.test_results.append(("Full Pipeline", "PASSED"))
            else:
                print(f"❌ Full Pipeline Failed: {response.status_code}")
                print(f"   Response: {response.text}")
                self.test_results.append(("Full Pipeline", f"FAILED: {response.status_code}"))
        except Exception as e:
            print(f"❌ Full Pipeline Error: {e}")
            self.test_results.append(("Full Pipeline", f"ERROR: {e}"))
    
    def run_all_tests(self):
        """Run all API tests"""
        print("=" * 60)
        print("API ENDPOINT TESTS")
        print("=" * 60)
        
        # Wait for services to be ready
        print("⏳ Waiting for services to be ready...")
        time.sleep(3)
        
        # Run tests
        self.test_health_check()
        
        # Check if any service is available
        try:
            health_response = requests.get(f"{self.base_url}/health", timeout=5)
            if health_response.status_code != 200:
                print("\n❌ API Gateway is not responding. Make sure services are running.")
                return
        except:
            print("\n❌ Cannot connect to API Gateway. Make sure services are running.")
            print("   Run: python start_enhanced_services.py")
            return
        
        # Continue with other tests
        self.test_quality_analysis()
        self.test_preprocessing()
        self.test_entity_extraction()
        self.test_full_pipeline()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, result in self.test_results if result == "PASSED")
        failed = len(self.test_results) - passed
        
        for test_name, result in self.test_results:
            status = "✅" if result == "PASSED" else "❌"
            print(f"{status} {test_name}: {result}")
        
        print("=" * 60)
        print(f"Total Tests: {len(self.test_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print("=" * 60)

def main():
    print("🧪 Enhanced Services API Test Suite\n")
    
    # Check if running in correct environment
    if not os.path.exists("microservices"):
        print("❌ Error: Must run from project root directory")
        return
    
    tester = APITester()
    tester.run_all_tests()
    
    print("\n💡 TIP: Visit http://localhost:8000/docs for interactive API documentation")

if __name__ == "__main__":
    main()
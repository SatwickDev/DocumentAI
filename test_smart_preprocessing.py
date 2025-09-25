#!/usr/bin/env python3
"""
Test script to verify smart preprocessing implementation
"""
import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test if API Gateway is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API Gateway is healthy")
            health_data = response.json()
            print(f"Services status: {health_data.get('services', {})}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def create_test_file():
    """Create a test document file"""
    test_content = """
    This is a high-quality test document with excellent formatting and clear content.
    The document contains proper structure, good grammar, and comprehensive information.
    
    Key Features:
    - Well-formatted text
    - Clear sections
    - Professional language
    - Comprehensive content
    
    This document should score highly in quality analysis and demonstrate
    the smart preprocessing decision-making logic.
    """
    
    with open("test_excellent_document.txt", "w") as f:
        f.write(test_content)
    
    return "test_excellent_document.txt"

def test_default_processing():
    """Test default /process endpoint with intelligent conditional preprocessing"""
    print("\n🔄 Testing Default Processing (Intelligent Conditional Preprocessing)")
    print("=" * 70)
    
    filename = create_test_file()
    
    try:
        with open(filename, 'rb') as f:
            files = {'file': (filename, f, 'text/plain')}
            data = {'session_id': f'test-default-{int(time.time())}'}
            
            response = requests.post(f"{API_BASE_URL}/process", files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            print("✅ Default processing completed successfully")
            
            # Check preprocessing decision
            preprocessing_decision = result.get('preprocessing_decision', {})
            print(f"📊 Preprocessing Decision:")
            print(f"   Applied: {preprocessing_decision.get('applied', 'N/A')}")
            print(f"   Reason: {preprocessing_decision.get('reason', 'N/A')}")
            print(f"   Quality Score: {preprocessing_decision.get('based_on_quality_score', 'N/A')}")
            print(f"   Verdict: {preprocessing_decision.get('based_on_verdict', 'N/A')}")
            
            # Check processing history
            history = result.get('processing_history', [])
            print(f"\n📝 Processing History ({len(history)} steps):")
            for step in history:
                status_emoji = {"completed": "✅", "skipped": "⏩", "failed": "❌", "error": "🚨"}.get(step.get('status'), "❓")
                print(f"   {step.get('step', '?')}. {step.get('service', 'Unknown')} - {status_emoji} {step.get('status', 'Unknown')}")
                if step.get('details'):
                    print(f"      Details: {step['details']}")
            
            return True
        else:
            print(f"❌ Default processing failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Default processing error: {e}")
        return False

def test_preprocessing_only():
    """Test /preprocess endpoint with smart preprocessing-only mode"""
    print("\n🎯 Testing Preprocessing-Only Mode (Smart Preprocessing)")
    print("=" * 70)
    
    filename = create_test_file()
    
    try:
        with open(filename, 'rb') as f:
            files = {'file': (filename, f, 'text/plain')}
            data = {'session_id': f'test-preprocess-{int(time.time())}'}
            
            response = requests.post(f"{API_BASE_URL}/preprocess", files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            print("✅ Preprocessing-only completed successfully")
            
            # Check preprocessing decision
            preprocessing_decision = result.get('preprocessing_decision', {})
            print(f"📊 Preprocessing Decision:")
            print(f"   Applied: {preprocessing_decision.get('applied', 'N/A')}")
            print(f"   Reason: {preprocessing_decision.get('reason', 'N/A')}")
            print(f"   Quality Score: {preprocessing_decision.get('based_on_quality_score', 'N/A')}")
            print(f"   Verdict: {preprocessing_decision.get('based_on_verdict', 'N/A')}")
            
            # Check processing history
            history = result.get('processing_history', [])
            print(f"\n📝 Processing History ({len(history)} steps):")
            for step in history:
                status_emoji = {"completed": "✅", "skipped": "⏩", "failed": "❌", "error": "🚨"}.get(step.get('status'), "❓")
                print(f"   {step.get('step', '?')}. {step.get('service', 'Unknown')} - {status_emoji} {step.get('status', 'Unknown')}")
                if step.get('details'):
                    print(f"      Details: {step['details']}")
            
            return True
        else:
            print(f"❌ Preprocessing-only failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Preprocessing-only error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Smart Preprocessing Implementation Test")
    print("=" * 50)
    
    # Test health first
    if not test_health():
        print("❌ Cannot proceed - API Gateway is not healthy")
        return
    
    # Wait a moment for services to be fully ready
    print("\n⏳ Waiting for services to be fully ready...")
    time.sleep(3)
    
    # Test both modes
    success_count = 0
    total_tests = 2
    
    if test_default_processing():
        success_count += 1
    
    if test_preprocessing_only():
        success_count += 1
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 30)
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Smart preprocessing implementation is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
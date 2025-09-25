#!/usr/bin/env python3
"""
Test preprocessing-only mode with a proper image file
"""
import requests
import time

def test_preprocessing_only_with_image():
    """Test preprocessing-only endpoint with an image file"""
    api_url = "http://localhost:8000"
    
    # Check if the image exists
    try:
        with open("test_document.png", "rb") as f:
            files = {"file": ("test_document.png", f, "image/png")}
            data = {"session_id": f"test-preprocess-img-{int(time.time())}"}
            
            print("🧪 Testing Preprocessing-Only Mode with Image File")
            print("=" * 60)
            print(f"📁 File: test_document.png")
            print(f"🔗 Endpoint: {api_url}/preprocess")
            print("⏳ Sending request...\n")
            
            response = requests.post(f"{api_url}/preprocess", files=files, data=data)
            
            print(f"📊 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ SUCCESS: Preprocessing-only completed!")
                
                # Show key information
                print(f"\n📋 Results Summary:")
                print(f"   📁 Filename: {result.get('filename', 'N/A')}")
                print(f"   💾 File Size: {result.get('file_size_mb', 'N/A'):.2f} MB")
                print(f"   ⏱️  Processing Time: {result.get('processing_time_seconds', 'N/A'):.2f}s")
                print(f"   ✅ Success: {result.get('success', 'N/A')}")
                
                # Show preprocessing decision
                if 'preprocessing_decision' in result:
                    decision = result['preprocessing_decision']
                    print(f"\n🧠 Preprocessing Decision:")
                    print(f"   Applied: {decision.get('applied', 'N/A')}")
                    print(f"   Reason: {decision.get('reason', 'N/A')}")
                    print(f"   Quality Score: {decision.get('based_on_quality_score', 'N/A')}")
                    print(f"   Verdict: {decision.get('based_on_verdict', 'N/A')}")
                
                # Show processing history
                if 'processing_history' in result:
                    history = result['processing_history']
                    print(f"\n📝 Processing History ({len(history)} steps):")
                    for step in history:
                        status_emoji = {
                            "completed": "✅", 
                            "skipped": "⏩", 
                            "failed": "❌", 
                            "error": "🚨"
                        }.get(step.get('status'), "❓")
                        
                        print(f"   {step.get('step', '?')}. {step.get('service', 'Unknown')} - {status_emoji} {step.get('status', 'Unknown')}")
                        if step.get('details'):
                            print(f"      💡 {step['details']}")
                        if step.get('error'):
                            print(f"      ⚠️  {step['error']}")
                
                # Show detailed preprocessing operations if available
                if 'preprocessing' in result and isinstance(result['preprocessing'], dict):
                    preprocessing = result['preprocessing']
                    if 'operations_applied' in preprocessing and preprocessing['operations_applied']:
                        print(f"\n🔧 Preprocessing Operations Applied:")
                        for op in preprocessing['operations_applied']:
                            print(f"   ✅ {op}")
                    
                    if 'improvements' in preprocessing and preprocessing['improvements']:
                        print(f"\n📈 Improvements Made:")
                        for improvement in preprocessing['improvements']:
                            print(f"   💡 {improvement}")
                    
                    if 'image_analysis' in preprocessing:
                        analysis = preprocessing['image_analysis']
                        print(f"\n📊 Image Analysis:")
                        print(f"   Original Contrast: {analysis.get('original_contrast', 'N/A')}")
                        print(f"   Original Brightness: {analysis.get('original_brightness', 'N/A')}")
                        print(f"   Deskewed: {analysis.get('deskewed', 'N/A')}")
                
                # Show any errors
                if 'errors' in result:
                    print(f"\n⚠️  Errors encountered:")
                    for service, error in result['errors'].items():
                        print(f"   {service}: {error}")
                
                print(f"\n🎉 Test completed successfully!")
                return True
                
            else:
                print(f"❌ FAILED: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except FileNotFoundError:
        print("❌ Error: test_document.png not found. Please run create_test_image.py first.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_preprocessing_only_with_image()
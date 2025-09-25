import requests

def test_preprocessing_with_json_format():
    """Test preprocessing service with return_format=base64"""
    print("🧪 Testing Preprocessing Service with JSON Format")
    print("=" * 60)
    
    url = "http://localhost:8003/preprocess"
    
    try:
        # Test with the image file and return_format=base64
        with open("test_document.png", "rb") as f:
            files = {"file": ("test_document.png", f, "image/png")}
            data = {"return_format": "base64"}
            
            print("📁 File: test_document.png")
            print("🔗 URL:", url)
            print("📋 Data: return_format=base64")
            print("⏳ Sending request...")
            
            response = requests.post(url, files=files, data=data)
            
            print(f"📊 Response Status: {response.status_code}")
            print(f"📊 Content-Type: {response.headers.get('content-type', 'Unknown')}")
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    result = response.json()
                    print("✅ SUCCESS: Got JSON response!")
                    print("📋 Operations Applied:", result.get('operations_applied', []))
                    print("📋 Improvements:", result.get('improvements', []))
                    print("📋 Status:", result.get('status', 'Unknown'))
                    if 'image_analysis' in result:
                        analysis = result['image_analysis']
                        print("📋 Image Analysis:")
                        print(f"   Original Contrast: {analysis.get('original_contrast', 'N/A')}")
                        print(f"   Original Brightness: {analysis.get('original_brightness', 'N/A')}")
                        print(f"   Deskewed: {analysis.get('deskewed', 'N/A')}")
                else:
                    print("❌ Still getting binary response instead of JSON")
                    print(f"   Content-Type: {content_type}")
                    print(f"   Response Size: {len(response.content)} bytes")
            else:
                print(f"❌ FAILED: {response.status_code}")
                print(f"Error: {response.text}")
                    
    except Exception as e:
        print(f"🚨 Exception: {str(e)}")
    
    print("🎉 Test completed!")

if __name__ == "__main__":
    test_preprocessing_with_json_format()
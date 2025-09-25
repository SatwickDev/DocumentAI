import requests
import time

def test_preprocessing_service_direct():
    """Test preprocessing service directly"""
    print("🧪 Testing Preprocessing Service Directly")
    print("=" * 60)
    
    url = "http://localhost:8003/preprocess"
    
    try:
        # Test with the image file
        with open("test_document.png", "rb") as f:
            files = {"file": ("test_document.png", f, "image/png")}
            
            print(f"📁 File: test_document.png")
            print(f"🔗 URL: {url}")
            print("⏳ Sending direct request to preprocessing service...")
            
            response = requests.post(url, files=files)
            
            print(f"📊 Response Status: {response.status_code}")
            print(f"📊 Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                print("✅ SUCCESS: Preprocessing service responded successfully!")
                
                # Check if response is JSON or binary
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    result = response.json()
                    print("📋 JSON Response:")
                    for key, value in result.items():
                        if key != 'data':  # Don't print base64 data
                            print(f"   {key}: {value}")
                else:
                    print(f"📋 Binary Response (Content-Type: {content_type})")
                    print(f"   Response Size: {len(response.content)} bytes")
                    
            else:
                print(f"❌ FAILED: {response.status_code}")
                try:
                    print(f"Error: {response.text}")
                except:
                    print("Error: Cannot decode response as text")
                    
    except Exception as e:
        print(f"🚨 Exception: {str(e)}")
    
    print("🎉 Direct test completed!")

if __name__ == "__main__":
    test_preprocessing_service_direct()
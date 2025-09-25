#!/usr/bin/env python3
import requests
import sys

try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"✅ API Gateway is running on port 8000")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except requests.ConnectionError:
    print("❌ Cannot connect to port 8000")
    print("Make sure the service is running")
except Exception as e:
    print(f"❌ Error: {e}")
#!/usr/bin/env python3
"""Test running services"""

import requests
import time

services = {
    "API Gateway": "http://localhost:8000/health",
    "Classification Service": "http://localhost:8001/health", 
    "Quality Service": "http://localhost:8002/health",
    "Preprocessing Service": "http://localhost:8003/health",
    "Entity Extraction Service": "http://localhost:8004/health",
    "Frontend": "http://localhost:8080/health"
}

print("Testing services...")
print("-" * 50)

for name, url in services.items():
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            print(f"✅ {name}: RUNNING at {url.replace('/health', '')}")
        else:
            print(f"⚠️  {name}: Responded with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"❌ {name}: NOT RUNNING or not accessible")
    except requests.exceptions.Timeout:
        print(f"⏱️  {name}: TIMEOUT")
    except Exception as e:
        print(f"❌ {name}: ERROR - {str(e)}")

print("-" * 50)
print("\nNote: Services running in WSL may need to be accessed from Windows browser")
print("Try opening in your Windows browser:")
print("- Frontend: http://localhost:8080")
print("- API Gateway Docs: http://localhost:8000/docs")
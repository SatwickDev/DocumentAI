#!/usr/bin/env python3
"""Test services on Windows"""

import requests
import time
import sys

def test_service(name, url):
    """Test a service endpoint"""
    try:
        print(f"\nTesting {name} at {url}...")
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✓ {name} is running")
            try:
                data = response.json()
                if 'status' in data:
                    print(f"  Status: {data['status']}")
                if 'services' in data:
                    print(f"  Connected services: {list(data['services'].keys())}")
            except:
                pass
            return True
        else:
            print(f"✗ {name} returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ {name} is not accessible (Connection refused)")
        return False
    except requests.exceptions.Timeout:
        print(f"✗ {name} timed out")
        return False
    except Exception as e:
        print(f"✗ {name} error: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("F2 Document Processing Services Test")
    print("=" * 60)
    
    services = [
        ("API Gateway", "http://localhost:8000/health"),
        ("Classification Service", "http://localhost:8001/health"),
        ("Quality Service", "http://localhost:8002/health"),
        ("Preprocessing Service", "http://localhost:8003/health"),
        ("Entity Extraction Service", "http://localhost:8004/health"),
        ("Frontend", "http://localhost:8080/health")
    ]
    
    results = []
    for name, url in services:
        results.append(test_service(name, url))
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    running = sum(results)
    total = len(results)
    
    if running == total:
        print(f"✓ All {total} services are running!")
    else:
        print(f"⚠ Only {running}/{total} services are running")
        print("\nTo start missing services, run:")
        print("  start_missing_services.bat")
        print("\nOr to restart all services:")
        print("  restart_services.bat")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
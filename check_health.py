#!/usr/bin/env python
"""
Script to check the health of all microservices running in Docker.
"""
import subprocess
import sys
import requests
import time
from concurrent.futures import ThreadPoolExecutor

# Define service endpoints
SERVICES = {
    "API Gateway": "http://localhost:8000/health",
    "Classification Service": "http://localhost:8001/health",
    "Quality Service": "http://localhost:8002/health",
    "Frontend": "http://localhost:80"
}

def check_docker_status():
    """Check if all containers are running."""
    try:
        result = subprocess.run(["docker-compose", "ps"], check=True, capture_output=True, text=True)
        print("Docker Container Status:")
        print(result.stdout)
        return True
    except subprocess.SubprocessError as e:
        print(f"Error checking Docker status: {e}")
        return False

def check_service_health(name, url):
    """Check if a service is healthy by making an HTTP request."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return name, True, f"Status: {response.status_code}"
        else:
            return name, False, f"Status: {response.status_code}"
    except requests.RequestException as e:
        return name, False, f"Error: {str(e)}"

def check_all_services():
    """Check health of all services in parallel."""
    print("Checking health of all services...")
    
    results = []
    with ThreadPoolExecutor(max_workers=len(SERVICES)) as executor:
        futures = {executor.submit(check_service_health, name, url): name for name, url in SERVICES.items()}
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                name = futures[future]
                results.append((name, False, f"Error: {str(e)}"))
    
    # Print results
    healthy_count = 0
    print("\nService Health Status:")
    for name, is_healthy, message in results:
        status = "✅ HEALTHY" if is_healthy else "❌ UNHEALTHY"
        if is_healthy:
            healthy_count += 1
        print(f"{name}: {status} - {message}")
    
    print(f"\n{healthy_count}/{len(SERVICES)} services are healthy")
    return healthy_count == len(SERVICES)

def main():
    """Main function to check health of Docker services."""
    if not check_docker_status():
        sys.exit(1)
    
    # Wait a bit for services to be fully up
    print("Waiting for services to initialize...")
    time.sleep(5)
    
    if not check_all_services():
        print("\nSome services are not healthy. Check the logs for more details:")
        print("Run: docker-compose logs -f")
        sys.exit(1)
    else:
        print("\nAll services are healthy and ready!")

if __name__ == "__main__":
    main()

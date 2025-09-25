#!/usr/bin/env python3
"""
Docker Services Health Check
Checks the health of all F2 services running in Docker
"""

import requests
import time
import subprocess
import json

def check_docker_running():
    """Check if Docker is running"""
    try:
        result = subprocess.run(['docker', 'version'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def get_container_status():
    """Get status of all containers"""
    try:
        result = subprocess.run(
            ['docker', 'ps', '--format', 'json'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    containers.append(json.loads(line))
            return containers
        return []
    except:
        return []

def check_service_health(name, url, timeout=5):
    """Check if a service is healthy"""
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        if response.status_code == 200:
            return True, "Healthy", response.json()
        else:
            return False, f"Unhealthy (Status: {response.status_code})", None
    except requests.exceptions.ConnectionError:
        return False, "Connection Failed", None
    except requests.exceptions.Timeout:
        return False, "Timeout", None
    except Exception as e:
        return False, f"Error: {str(e)}", None

def main():
    print("=" * 60)
    print("F2 Document Processing - Docker Health Check")
    print("=" * 60)
    
    # Check Docker
    if not check_docker_running():
        print("❌ Docker is not running! Please start Docker Desktop.")
        return
    
    print("✅ Docker is running\n")
    
    # Get container status
    containers = get_container_status()
    print("📦 Running Containers:")
    for container in containers:
        if 'f2-' in container.get('Names', ''):
            print(f"  - {container['Names']}: {container['Status']}")
    
    print("\n🏥 Service Health Checks:")
    print("-" * 60)
    
    services = {
        "API Gateway": "http://localhost:8000",
        "Classification Service": "http://localhost:8001",
        "Quality Service": "http://localhost:8002",
        "Preprocessing Service": "http://localhost:8003",
        "Entity Extraction Service": "http://localhost:8004",
        "Frontend": "http://localhost:8080"
    }
    
    all_healthy = True
    
    for name, base_url in services.items():
        is_healthy, status, health_data = check_service_health(name, base_url)
        
        status_icon = "✅" if is_healthy else "❌"
        print(f"{status_icon} {name:25} - {status}")
        
        if health_data and 'services' in health_data:
            # For API Gateway, show connected services
            print(f"   Connected Services:")
            for service, is_connected in health_data['services'].items():
                conn_icon = "✓" if is_connected else "✗"
                print(f"     {conn_icon} {service}")
        
        if not is_healthy:
            all_healthy = False
    
    print("-" * 60)
    
    if all_healthy:
        print("\n✅ All services are healthy!")
        print("\n🌐 You can access:")
        print("   - Frontend: http://localhost:8080")
        print("   - API Docs: http://localhost:8000/docs")
    else:
        print("\n⚠️  Some services are not healthy.")
        print("\nTroubleshooting:")
        print("1. Check logs: docker-compose -f docker-compose.simple.yml logs")
        print("2. Restart services: docker-compose -f docker-compose.simple.yml restart")
        print("3. Rebuild: docker-compose -f docker-compose.simple.yml down && docker-compose -f docker-compose.simple.yml up -d")

if __name__ == "__main__":
    main()
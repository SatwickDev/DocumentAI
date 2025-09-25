import requests
import sys

def check_service(name, url):
    try:
        response = requests.get(url, timeout=5)
        print(f"{name}: {response.status_code} - {'OK' if response.status_code == 200 else 'Error'}")
        return True
    except Exception as e:
        print(f"{name}: Error - {str(e)}")
        return False

services = {
    "API Gateway": "http://localhost:8000/docs",
    "Classification Service": "http://localhost:8001/docs",
    "Quality Service": "http://localhost:8002/docs",
    "MCP Orchestrator": "http://localhost:8003/docs",
    "Notification Service": "http://localhost:8004/docs",
    "Frontend": "http://localhost:8080"
}

all_ok = True
for name, url in services.items():
    if not check_service(name, url):
        all_ok = False

sys.exit(0 if all_ok else 1)

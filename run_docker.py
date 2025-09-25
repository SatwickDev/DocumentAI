#!/usr/bin/env python
"""
Script to run all microservices using Docker Compose.
"""
import os
import subprocess
import sys
import time

def check_docker_installed():
    """Check if Docker and Docker Compose are installed."""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: Docker or Docker Compose not installed or not in PATH.")
        print("Please install Docker Desktop from https://www.docker.com/products/docker-desktop")
        return False

def run_docker_compose():
    """Run docker-compose up command."""
    try:
        print("Starting all microservices with Docker Compose...")
        subprocess.run(["docker-compose", "up", "--build", "-d"], check=True)
        
        print("\nServices are starting up. Checking status...")
        time.sleep(10)  # Give services time to start
        
        # Check if services are running
        result = subprocess.run(["docker-compose", "ps"], check=True, capture_output=True, text=True)
        print("\nService Status:")
        print(result.stdout)
        
        print("\nAll microservices are now running!")
        print("API Gateway is available at: http://localhost:8000")
        print("Frontend is available at: http://localhost:80")
        print("\nTo view logs, run: docker-compose logs -f")
        print("To stop all services, run: docker-compose down")
        
        return True
    except subprocess.SubprocessError as e:
        print(f"Error running Docker Compose: {e}")
        return False

def main():
    """Main function to run Docker Compose services."""
    if not check_docker_installed():
        sys.exit(1)
    
    if not run_docker_compose():
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Script to stop all microservices running in Docker Compose.
"""
import subprocess
import sys

def stop_docker_compose():
    """Stop docker-compose services."""
    try:
        print("Stopping all microservices...")
        subprocess.run(["docker-compose", "down"], check=True)
        print("All microservices have been stopped.")
        return True
    except subprocess.SubprocessError as e:
        print(f"Error stopping Docker Compose: {e}")
        return False

def main():
    """Main function to stop Docker Compose services."""
    if not stop_docker_compose():
        sys.exit(1)

if __name__ == "__main__":
    main()

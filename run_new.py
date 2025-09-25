#!/usr/bin/env python3
"""
MCP Document Processing System - Main Runner
Standardized entry point for all services and utilities
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def start_classification_service(port=8001):
    """Start the classification microservice"""
    print(f"🚀 Starting Classification Service on port {port}...")
    print(f"   🌐 URL: http://localhost:{port}")
    print(f"   📖 Docs: http://localhost:{port}/docs")
    print()
    
    # Run the service using module path
    subprocess.run([
        sys.executable, "-m", "src.microservices.classification_service"
    ])

def start_quality_service(port=8002):
    """Start the quality analysis microservice"""
    print(f"🚀 Starting Quality Analysis Service on port {port}...")
    print(f"   🌐 URL: http://localhost:{port}")
    print(f"   📖 Docs: http://localhost:{port}/docs")
    print()
    
    # Run the service using module path
    subprocess.run([
        sys.executable, "-m", "src.microservices.quality_service"
    ])

def start_api_gateway(port=8000):
    """Start the API Gateway"""
    print(f"🚀 Starting API Gateway on port {port}...")
    print(f"   🌐 URL: http://localhost:{port}")
    print(f"   📖 Docs: http://localhost:{port}/docs")
    print()
    
    # Run the API gateway using module path
    subprocess.run([
        sys.executable, "-m", "src.microservices.api_gateway"
    ])

def start_mcp_orchestrator():
    """Start the MCP Orchestrator"""
    print(f"🚀 Starting MCP Orchestrator...")
    
    # Run the orchestrator using module path
    subprocess.run([
        sys.executable, "-m", "src.utils.mcp_orchestrator"
    ])

def start_notification_service():
    """Start the Notification Service"""
    print(f"🚀 Starting Notification Service...")
    
    # Run the service using module path
    subprocess.run([
        sys.executable, "-m", "src.microservices.notification_service"
    ])

def start_frontend():
    """Start a simple HTTP server for the frontend"""
    frontend_path = Path(__file__).parent / "frontend"
    
    print("🚀 Starting Frontend Server...")
    print(f"   📁 Path: {frontend_path}")
    print("   🌐 URL: http://localhost:8080")
    print()
    
    # Change to the frontend directory
    os.chdir(frontend_path)
    
    # Start a simple HTTP server
    subprocess.run([
        sys.executable, "-m", "http.server", "8080"
    ])

def start_mcp_server(server_type="classification"):
    """Start an MCP server"""
    print(f"🚀 Starting {server_type.title()} MCP Server...")
    
    if server_type.lower() == "classification":
        # Run the classification MCP server
        subprocess.run([
            sys.executable, "-m", "src.mcp_servers.classification_mcp_server"
        ])
    elif server_type.lower() == "quality":
        # Run the quality MCP server
        subprocess.run([
            sys.executable, "-m", "src.mcp_servers.quality_mcp_server"
        ])
    else:
        print(f"❌ Unknown server type: {server_type}")
        print("   Supported types: classification, quality")

def run_tests():
    """Run the test suite"""
    tests_path = Path(__file__).parent / "tests"
    
    print("🧪 Running Tests...")
    print(f"   📁 Path: {tests_path}")
    print()
    
    # Run pytest
    subprocess.run([
        sys.executable, "-m", "pytest", "-v", tests_path
    ])

def show_project_structure():
    """Display the project structure"""
    print("📂 Project Structure:")
    print()
    
    # Use tree command if available, otherwise list directories
    try:
        subprocess.run(["tree", "--dirsfirst", "-L", "3", "-I", "__pycache__|*.pyc|venv|.git"])
    except FileNotFoundError:
        # Fallback to basic directory listing
        for root, dirs, files in os.walk(".", topdown=True):
            level = root.count(os.sep)
            indent = '│   ' * level
            print(f"{indent}├── {os.path.basename(root)}/")
            for file in files:
                if file.endswith(".pyc") or file == "__pycache__":
                    continue
                print(f"{indent}│   ├── {file}")

def run_docker_compose():
    """Start all services using Docker Compose"""
    docker_compose_path = Path(__file__).parent / "docker" / "docker-compose.yml"
    
    print("🐳 Starting Docker Compose Services...")
    print(f"   📁 Path: {docker_compose_path}")
    print()
    
    # Run docker-compose up
    subprocess.run([
        "docker-compose", "-f", str(docker_compose_path), "up"
    ])

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MCP Document Processing System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Service command
    service_parser = subparsers.add_parser("service", help="Start a microservice")
    service_parser.add_argument("service_type", nargs="?", default="classification", 
                                choices=["classification", "quality", "gateway"],
                                help="Type of service to start")
    
    # MCP server command
    mcp_parser = subparsers.add_parser("mcp", help="Start an MCP server")
    mcp_parser.add_argument("server_type", nargs="?", default="classification",
                           choices=["classification", "quality"],
                           help="Type of MCP server to start")
    
    # Frontend command
    subparsers.add_parser("frontend", help="Start the frontend server")
    
    # Test command
    subparsers.add_parser("test", help="Run tests")
    
    # Structure command
    subparsers.add_parser("structure", help="Show project structure")
    
    # Docker command
    subparsers.add_parser("docker", help="Start services with Docker Compose")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "service":
        if args.service_type == "classification":
            start_classification_service()
        elif args.service_type == "quality":
            start_quality_service()
        elif args.service_type == "gateway":
            start_api_gateway()
    elif args.command == "mcp":
        start_mcp_server(args.server_type)
    elif args.command == "frontend":
        start_frontend()
    elif args.command == "test":
        run_tests()
    elif args.command == "structure":
        show_project_structure()
    elif args.command == "docker":
        run_docker_compose()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

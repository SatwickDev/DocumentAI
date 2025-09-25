#!/usr/bin/env python3
"""
Complete Non-Docker Service Runner
Starts all MCP services, microservices, orchestrator, and frontend without Docker
"""

import os
import sys
import sys
sys.stdout.reconfigure(encoding='utf-8')
import subprocess

import time
import threading
import webbrowser
from pathlib import Path
import signal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set environment variables
os.environ['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), 'src')
os.environ['CONFIG_PATH'] = os.path.join(os.path.dirname(__file__), 'src', 'core', 'classification_config.json')
os.environ['TESSERACT_CMD'] = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['OUTPUT_BASE_PATH'] = os.path.join(os.path.dirname(__file__), 'output')

class ServiceManager:
    def __init__(self):
        self.processes = []
        self.services = {}
        
    def start_service(self, name, command, delay=10):
        """Start a service with environment setup"""
        print(f"[START] Starting {name}...")
        try:
            # Set up environment variables
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{os.path.join(os.path.dirname(__file__), 'src')}{os.pathsep}{env.get('PYTHONPATH', '')}"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"
            
            process = subprocess.Popen(
                command, 
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=os.path.dirname(__file__)
            )
            
            self.processes.append(process)
            self.services[name] = {
                'process': process,
                'command': command,
                'pid': process.pid
            }
            print(f"[OK] {name} started with PID: {process.pid}")
            
            # Wait for service to initialize
            retry_count = 0
            max_retries = 5
            while retry_count < max_retries:
                time.sleep(delay)
                # Check if process is still running
                if process.poll() is not None:
                    error = process.stderr.read().decode('utf-8')
                    print(f"[ERROR] {name} failed to start: {error}")
                    return False
                    
                # Check service health
                if self._check_service_health(process):
                    print(f"[OK] {name} initialized successfully")
                    return True
                    
                retry_count += 1
                print(f"[WAIT] Waiting for {name} to initialize (attempt {retry_count}/{max_retries})")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to start {name}: {e}")
            return False
    
    def _check_service_health(self, process):
        """Check if a service is healthy and fully initialized"""
        try:
            # Check if process is running
            if process.poll() is not None:
                logger.error(f"Process terminated with exit code {process.poll()}")
                return False
                
            # Read any error output
            stderr_data = process.stderr.peek().decode().strip()
            if stderr_data:
                logger.error(f"Process error output: {stderr_data}")
                return False
                
            # Read stdout for initialization messages
            stdout_data = process.stdout.peek().decode().strip()
            if "ready for connections" in stdout_data.lower() or "server initialization complete" in stdout_data.lower():
                return True
                
            logger.info(f"Service output: {stdout_data}")
            return False
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

def run_services():
    """Run all services in the correct order"""
    print("[START] Starting MCP Document Processing System (Non-Docker)")
    print("=" * 60)

    # Initialize service manager
    manager = ServiceManager()
    
    print("[INFO] Starting MCP Servers...")
    
    # Start Classification MCP Server
    print("[START] Starting Classification MCP Server...")
    if not manager.start_service(
        "Classification MCP Server",
        [sys.executable, "-m", "src.mcp_servers.classification_mcp_server"]
    ):
        print("[ERROR] Failed to start Classification MCP Server")
        return False
        
    # Start Quality MCP Server
    print("[START] Starting Quality MCP Server...")
    if not manager.start_service(
        "Quality MCP Server",
        [sys.executable, "-m", "src.mcp_servers.quality_mcp_server"]
    ):
        print("[ERROR] Failed to start Quality MCP Server")
        return False
    
    # Start MCP Orchestrator
    print("[INFO] Starting MCP Orchestrator...")
    print("[START] Starting MCP Orchestrator...")
    if not manager.start_service(
        "MCP Orchestrator",
        [sys.executable, "-m", "src.utils.mcp_orchestrator"]
    ):
        print("[ERROR] Failed to start MCP Orchestrator")
        return False
    
    # Start Microservices
    print("[INFO] Starting Microservices...")
    
    # Start Classification Service
    print("[START] Starting Classification Service...")
    if not manager.start_service(
        "Classification Service",
        [sys.executable, "-m", "uvicorn", "src.microservices.classification_service:app", "--host", "0.0.0.0", "--port", "8001"]
    ):
        print("[ERROR] Failed to start Classification Service")
        return False
    
    # Start Quality Service
    print("[START] Starting Quality Service...")
    if not manager.start_service(
        "Quality Service",
        [sys.executable, "-m", "uvicorn", "src.microservices.quality_service:app", "--host", "0.0.0.0", "--port", "8002"]
    ):
        print("[ERROR] Failed to start Quality Service")
        return False
    
    # Start API Gateway
    print("[START] Starting API Gateway...")
    if not manager.start_service(
        "API Gateway",
        [sys.executable, "-m", "uvicorn", "src.microservices.api_gateway:app", "--host", "0.0.0.0", "--port", "8000"]
    ):
        print("[ERROR] Failed to start API Gateway")
        return False
    
    # Start Frontend
    print("[INFO] Starting Frontend...")
    print("[START] Starting Frontend Server...")
    if not manager.start_service(
        "Frontend Server",
        [sys.executable, "-m", "uvicorn", "src.serve_frontend:app", "--host", "0.0.0.0", "--port", "8080"]
    ):
        print("[ERROR] Failed to start Frontend Server")
        return False
    
    print("\nWaiting for services to initialize...")
    
    # Print service URLs and running services
    print("\n[INFO] Service URLs:")
    print("   Frontend:              http://localhost:8080")
    print("   API Gateway:           http://localhost:8000")
    print("   API Docs:              http://localhost:8000/docs")
    print("   Classification Service: http://localhost:8001")
    print("   Classification Docs:    http://localhost:8001/docs")
    
    print("\n[INFO] Running Services:")
    for name, service in manager.services.items():
        print(f"   • {name} (PID: {service['pid']})")
    
    print("\n[INFO] Tips:")
    print("   • Press Ctrl+C to stop all services")
    print("   • Use 'python check_health.py' to check service health")
    print("   • Frontend will open automatically in your browser")
    
    print("\nServices are running. Press Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Clean shutdown on Ctrl+C
        print("\n[INFO] Shutting down services...")
        for process in manager.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        print("[INFO] All services stopped")
        
if __name__ == "__main__":
    run_services()

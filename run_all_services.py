#!/usr/bin/env python3
"""
Complete Non-Docker Service Runner
Starts all MCP services, microservices, orchestrator, and frontend without Docker
"""

import os
import sys
import subprocess
import time
import threading
import webbrowser
from pathlib import Path
import signal
import logging

# Configure Windows console for UTF-8 output
if sys.platform == "win32":
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleOutputCP(65001)  # Set console to UTF-8

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
        print(f"🚀 Starting {name}...")
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
            print(f"   ✅ {name} started with PID: {process.pid}")
            
            # Wait for service to initialize
            retry_count = 0
            max_retries = 5
            while retry_count < max_retries:
                time.sleep(delay)
                # Check if process is still running
                if process.poll() is not None:
                    error = process.stderr.read().decode('utf-8')
                    print(f"   ❌ {name} failed to start: {error}")
                    return False
                    
                # Check service health
                if self._check_service_health(process):
                    print(f"   ✅ {name} initialized successfully")
                    return True
                    
                retry_count += 1
                print(f"   ⏳ Waiting for {name} to initialize (attempt {retry_count}/{max_retries})")
            return True
        except Exception as e:
            print(f"   ❌ Failed to start {name}: {e}")
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
            logger.error(f"Health check error: {e}")
            return False
        
    def stop_all_services(self):
        """Stop all running services"""
        print("\n🛑 Stopping all services...")
        for name, service in self.services.items():
            try:
                process = service['process']
                process.terminate()
                process.wait(timeout=5)
                print(f"   ✅ Stopped {name}")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"   ⚠️  Force killed {name}")
            except Exception as e:
                print(f"   ❌ Error stopping {name}: {e}")
    
    def check_service_health(self, name, url, timeout=5):
        """Check if a service is healthy"""
        import requests
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                print(f"   ✅ {name} is healthy")
                return True
        except:
            pass
        print(f"   ⚠️  {name} health check failed")
        return False

def run_all_services():
    """Run all services without Docker in the correct order"""
    manager = ServiceManager()
    
    print("🌟 Starting MCP Document Processing System (Non-Docker)")
    print("=" * 60)
    
    # Get Python executable path
    python_exe = sys.executable
    project_root = os.path.dirname(__file__)
    
    try:
        # 1. Start MCP Servers first
        print("\n📡 Starting MCP Servers...")
        
        # Classification MCP Server
        if not manager.start_service(
            "Classification MCP Server",
            [python_exe, os.path.join(project_root, "src", "mcp_servers", "classification_mcp_server.py")],
            delay=3
        ):
            print("❌ Failed to start Classification MCP Server")
            return False
        
        # Quality MCP Server
        if not manager.start_service(
            "Quality MCP Server", 
            [python_exe, os.path.join(project_root, "src", "mcp_servers", "quality_mcp_server.py")],
            delay=3
        ):
            print("❌ Failed to start Quality MCP Server")
            return False
        
        # Enhanced Quality MCP Server
        if not manager.start_service(
            "Enhanced Quality MCP Server",
            [python_exe, os.path.join(project_root, "src", "mcp_servers", "enhanced_quality_mcp_server.py")],
            delay=3
        ):
            print("❌ Failed to start Enhanced Quality MCP Server")
            # Don't fail if enhanced server doesn't start
        
        # Preprocessing MCP Server - removed, now using microservice
        
        # Entity Extraction MCP Server
        if not manager.start_service(
            "Entity Extraction MCP Server",
            [python_exe, os.path.join(project_root, "src", "mcp_servers", "entity_extraction_mcp_server.py")],
            delay=3
        ):
            print("❌ Failed to start Entity Extraction MCP Server")
            # Don't fail if entity extraction server doesn't start
        
        # 2. Start Microservices directly
        print("\n�️  Starting Microservices...")
        
        # API Gateway
        if not manager.start_service(
            "API Gateway",
            [python_exe, os.path.join(project_root, "microservices", "api-gateway", "app.py")],
            delay=3
        ):
            print("❌ Failed to start API Gateway")
            return False
        
        # Classification Service
        if not manager.start_service(
            "Classification Service",
            [python_exe, os.path.join(project_root, "microservices", "classification-service", "app.py")],
            delay=3
        ):
            print("❌ Failed to start Classification Service")
            return False
        
        # Quality Service
        if not manager.start_service(
            "Quality Service",
            [python_exe, os.path.join(project_root, "microservices", "quality-service", "app.py")],
            delay=3
        ):
            print("❌ Failed to start Quality Service")
            return False
        
        # Preprocessing Service (New)
        if not manager.start_service(
            "Preprocessing Service",
            [python_exe, os.path.join(project_root, "microservices", "preprocessing-service", "app.py")],
            delay=3
        ):
            print("❌ Failed to start Preprocessing Service")
            # Don't fail if preprocessing service doesn't start
        
        # Entity Extraction Service (New)
        if not manager.start_service(
            "Entity Extraction Service",
            [python_exe, os.path.join(project_root, "microservices", "entity-extraction-service", "app.py")],
            delay=3
        ):
            print("❌ Failed to start Entity Extraction Service")
            # Don't fail if entity extraction service doesn't start
        
        # 4. Start Frontend with FastAPI
        print("\n🌐 Starting Frontend...")
        if not manager.start_service(
            "Frontend",
            [python_exe, os.path.join(project_root, "serve_frontend.py")],
            delay=3
        ):
            print("❌ Failed to start Frontend")
            return False
        
        print("\n✅ All services started successfully!")
        print("\n🌐 Service URLs:")
        print("   API Gateway: http://localhost:8000")
        print("   Classification Service: http://localhost:8001")
        print("   Quality Service: http://localhost:8002")
        print("   Preprocessing Service: http://localhost:8003")
        print("   Entity Extraction Service: http://localhost:8004")
        print("   Frontend: http://localhost:8080")
        
        # 5. Wait and check health
        print("\n⏳ Waiting for services to initialize...")
        time.sleep(10)
        
        print("\n🔍 Checking Service Health...")
        health_urls = {
            "API Gateway": "http://localhost:8000/health",
            "Classification Service": "http://localhost:8001/health",
            "Quality Service": "http://localhost:8002/health",
            "Frontend": "http://localhost:8080"
        }
        
        for name, url in health_urls.items():
            try:
                manager.check_service_health(name, url)
            except:
                pass
        
        return True
        print("   🚪 API Gateway:           http://localhost:8000")
        print("   📊 API Docs:              http://localhost:8000/docs")
        print("   🔍 Classification Service: http://localhost:8001")
        print("   📖 Classification Docs:   http://localhost:8001/docs")
        print("\n📋 Running Services:")
        for name, service in manager.services.items():
            print(f"   • {name} (PID: {service['pid']})")
        
        print("\n💡 Tips:")
        print("   • Press Ctrl+C to stop all services")
        print("   • Use 'python check_health.py' to check service health")
        print("   • Frontend will open automatically in your browser")
        
        # Open browser
        try:
            webbrowser.open("http://localhost:8080")
        except:
            pass
        
        # Keep services running
        print("\n⏰ Services are running. Press Ctrl+C to stop...")
        
        # Set up signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\n\n🛑 Received interrupt signal...")
            manager.stop_all_services()
            print("✅ All services stopped. Goodbye!")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            signal_handler(None, None)
            
    except Exception as e:
        print(f"\n❌ Error starting services: {e}")
        manager.stop_all_services()
        sys.exit(1)

if __name__ == "__main__":
    run_all_services()

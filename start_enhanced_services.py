#!/usr/bin/env python3
"""
Startup script for enhanced services
Run this to start all the enhanced microservices
"""

import subprocess
import sys
import time
import os
import signal
import psutil

class ServiceManager:
    def __init__(self):
        self.processes = {}
        self.python_exe = "./venv/Scripts/python.exe" if os.name == 'nt' else "python3"
        
    def start_service(self, name, command, port):
        """Start a microservice"""
        print(f"🚀 Starting {name} on port {port}...")
        
        try:
            # Check if port is already in use
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    print(f"⚠️  Port {port} is already in use. Skipping {name}.")
                    return False
        except:
            pass
        
        try:
            process = subprocess.Popen(
                [self.python_exe] + command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes[name] = process
            print(f"✅ {name} started with PID {process.pid}")
            return True
        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return False
    
    def stop_all(self):
        """Stop all services"""
        print("\n🛑 Stopping all services...")
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} stopped")
            except:
                try:
                    process.kill()
                    print(f"⚠️  {name} force killed")
                except:
                    pass
    
    def run(self):
        """Run all services"""
        services = [
            # Existing services
            ("API Gateway", ["microservices/api-gateway/enhanced_app.py"], 8000),
            ("Classification Service", ["microservices/classification-service/app.py"], 8001),
            ("Quality Service", ["microservices/quality-service/app.py"], 8002),
            # New services
            ("Preprocessing Service", ["microservices/preprocessing-service/app.py"], 8003),
            ("Entity Extraction Service", ["microservices/entity-extraction-service/app.py"], 8004),
        ]
        
        print("=" * 60)
        print("ENHANCED MICROSERVICES STARTUP")
        print("=" * 60)
        
        # Start services
        started = 0
        for name, command, port in services:
            if self.start_service(name, command, port):
                started += 1
                time.sleep(2)  # Give service time to start
        
        print(f"\n✅ Started {started}/{len(services)} services")
        
        if started == 0:
            print("❌ No services started. Exiting.")
            return
        
        print("\n📋 AVAILABLE ENDPOINTS:")
        print("   - API Gateway: http://localhost:8000/docs")
        print("   - Full Pipeline: POST http://localhost:8000/process/full-pipeline")
        print("   - Quality Analysis: POST http://localhost:8000/analyze/quality")
        print("   - Preprocessing: POST http://localhost:8000/preprocess")
        print("   - Entity Extraction: POST http://localhost:8000/extract/entities")
        print("   - Classification + Entities: POST http://localhost:8000/classify/entities")
        
        print("\n💡 TIP: Use the API Gateway at http://localhost:8000/docs to test all endpoints")
        print("\nPress Ctrl+C to stop all services...")
        
        try:
            # Keep running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_all()

def main():
    manager = ServiceManager()
    
    # Handle signals
    def signal_handler(signum, frame):
        manager.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run services
    manager.run()

if __name__ == "__main__":
    main()
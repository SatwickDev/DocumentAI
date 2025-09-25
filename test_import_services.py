#!/usr/bin/env python3
"""Test service imports"""

import sys
import os

print("Testing Service Imports...")
print("=" * 50)

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'microservices', 'preprocessing-service'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'microservices', 'entity-extraction-service'))

# Test imports
print("\n1. Testing basic imports...")
try:
    import fastapi
    print("   ✓ FastAPI imported")
except ImportError:
    print("   ✗ FastAPI not installed")

try:
    import uvicorn
    print("   ✓ Uvicorn imported")
except ImportError:
    print("   ✗ Uvicorn not installed")

# Test service files
print("\n2. Checking service files...")
preprocessing_file = os.path.join("microservices", "preprocessing-service", "app_simple.py")
entity_file = os.path.join("microservices", "entity-extraction-service", "app_simple.py")

print(f"   Preprocessing service: {'EXISTS' if os.path.exists(preprocessing_file) else 'NOT FOUND'}")
print(f"   Entity extraction service: {'EXISTS' if os.path.exists(entity_file) else 'NOT FOUND'}")

print("\n" + "=" * 50)
print("Run start_enhanced_services.bat to start the services")
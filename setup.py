#!/usr/bin/env python3
"""
Setup script for Document Processing Microservices
Run this to install dependencies and set up the environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "output",
        "logs", 
        "temp",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def copy_config_files():
    """Ensure config files are in the right place"""
    config_files = [
        ("classification_config 3.json", "config/classification_config.json"),
        ("TresholdConfig.json", "config/TresholdConfig.json")
    ]
    
    for src, dst in config_files:
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
            print(f"✅ Copied {src} to {dst}")
        else:
            print(f"⚠️ Warning: {src} not found")

def fix_import_names():
    """Fix import names in the modules"""
    # Create proper module files with correct names
    import shutil
    
    files_to_rename = [
        ("documentClassifier 2.py", "documentClassifier.py"),
        ("QualityAnalysis 1.py", "QualityAnalysis.py")
    ]
    
    for old_name, new_name in files_to_rename:
        if os.path.exists(old_name) and not os.path.exists(new_name):
            shutil.copy2(old_name, new_name)
            print(f"✅ Created {new_name} from {old_name}")

def main():
    """Main setup function"""
    print("🚀 Document Processing Microservices Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Fix import names
    print("\n📝 Fixing module names...")
    fix_import_names()
    
    # Copy config files
    print("\n⚙️ Setting up configuration...")
    copy_config_files()
    
    # Install Python dependencies
    print("\n📦 Installing Python dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("⚠️ Some dependencies may have failed to install")
    
    # Check if Tesseract is available (for classification service)
    print("\n🔍 Checking Tesseract OCR...")
    if not run_command("tesseract --version", "Checking Tesseract"):
        print("⚠️ Tesseract OCR not found. Classification may not work properly.")
        print("Please install Tesseract OCR:")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("  macOS: brew install tesseract")
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_imports = [
        ("fastapi", "FastAPI framework"),
        ("uvicorn", "ASGI server"),
        ("httpx", "HTTP client"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("fitz", "PyMuPDF")
    ]
    
    failed_imports = []
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description} - Import failed")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️ Some imports failed: {', '.join(failed_imports)}")
        print("Try installing them manually with pip")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Start individual services:")
    print("   python classification_microservice.py  # Port 8001")
    print("   python quality_microservice.py         # Port 8002") 
    print("   python api_gateway.py                  # Port 8000")
    print("\n2. Or use Docker Compose:")
    print("   docker-compose up --build")
    print("\n3. Test the services:")
    print("   curl http://localhost:8000/health")
    print("   curl http://localhost:8000/api/v1/service-status")
    print("\n4. Access API documentation:")
    print("   http://localhost:8000/docs")
    
    if failed_imports:
        print(f"\n⚠️ Note: Fix the import issues first: {', '.join(failed_imports)}")

if __name__ == "__main__":
    main()

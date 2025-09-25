#!/usr/bin/env python3
"""
Final Cleanup Script
Run this script to complete the project structure cleanup
"""

import os
import shutil
from pathlib import Path
import sys

def main():
    """Main cleanup function"""
    # Get the project root
    project_root = Path(__file__).parent

    print("\n🧹 Final Cleanup Steps...\n")

    # 1. Replace run.py with the updated version
    print("\n📄 Replacing run.py with updated version...\n")
    run_py = project_root / "run.py"
    run_new_py = project_root / "run_new.py"
    
    if run_new_py.exists():
        # Backup the original
        backup_path = project_root / "run.py.bak"
        try:
            shutil.copy(run_py, backup_path)
            print(f"✅ Backed up original run.py to {backup_path}")
            
            # Replace with the new file
            shutil.copy(run_new_py, run_py)
            print(f"✅ Replaced run.py with updated version")
            
            # Remove the temporary file
            os.remove(run_new_py)
            print(f"✅ Removed temporary file {run_new_py}")
        except Exception as e:
            print(f"❌ Error during file replacement: {e}")
    else:
        print(f"❌ New run.py file not found at {run_new_py}")
    
    # 2. Check for empty requirements.txt and fix if needed
    print("\n📦 Checking requirements.txt...\n")
    req_file = project_root / "requirements.txt"
    
    if req_file.exists() and req_file.stat().st_size == 0:
        print("⚠️ requirements.txt is empty, restoring content...")
        try:
            with open(req_file, "w") as f:
                f.write("""# Requirements for Document Processing Microservices
# Install with: pip install -r requirements.txt

# Core dependencies for existing functionality
PyMuPDF==1.23.26            # fitz for PDF processing
pytesseract==0.3.10         # OCR capabilities
opencv-python==4.8.1.78     # Image processing
Pillow==10.1.0              # Image handling
numpy==1.24.3               # Numerical operations
pandas==2.1.4               # Data manipulation

# Document processing
python-docx==1.1.0          # DOCX support
openpyxl==3.1.2             # Excel support
xlrd==2.0.1                 # Excel reading

# Optimization
numba==0.58.1               # JIT compilation for performance

# Web framework and API
fastapi==0.104.1            # Modern web framework
uvicorn==0.24.0             # ASGI server
httpx==0.25.2               # Async HTTP client

# Development and testing
pytest==7.4.3              # Testing framework
pytest-asyncio==0.21.1     # Async testing support

# Utilities
python-multipart==0.0.6    # File upload support
aiofiles==23.2.1            # Async file operations

# Optional: For production deployment
gunicorn==21.2.0            # Production WSGI server
redis==5.0.1                # Caching and session storage

# Logging and monitoring
structlog==23.2.0           # Structured logging
""")
            print("✅ Restored requirements.txt content")
        except Exception as e:
            print(f"❌ Error writing to requirements.txt: {e}")
    else:
        print("✅ requirements.txt exists and has content")
    
    # 3. Remove cleanup scripts
    print("\n🗑️ Removing cleanup scripts...\n")
    cleanup_scripts = [
        project_root / "cleanup_project.py",
        project_root / "remove_unwanted_files.py",
        project_root / "final_cleanup.py",  # This file
    ]
    
    for script in cleanup_scripts:
        if script.exists():
            try:
                os.remove(script)
                print(f"✅ Removed {script}")
            except Exception as e:
                print(f"❌ Error removing {script}: {e}")
    
    print("\n✅ Final cleanup completed!\n")
    print("Your project structure is now clean and organized according to Python best practices.")
    print("You can now use the updated run.py script to manage your project:")
    print("  - python run.py service classification   # Start classification service")
    print("  - python run.py service quality          # Start quality service")
    print("  - python run.py service gateway          # Start API gateway")
    print("  - python run.py mcp classification       # Start classification MCP server")
    print("  - python run.py frontend                 # Start frontend server")
    print("  - python run.py test                     # Run tests")
    print("  - python run.py structure                # Show project structure")
    print("  - python run.py docker                   # Start with Docker Compose")

if __name__ == "__main__":
    main()

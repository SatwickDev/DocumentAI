#!/usr/bin/env python3
"""
Unwanted Files Cleanup Script
This script removes the unwanted files that should be deleted from the project
"""

import os
from pathlib import Path
import sys

def remove_file(file_path):
    """Remove a file if it exists"""
    try:
        if file_path.exists():
            os.remove(file_path)
            print(f"✅ Removed: {file_path}")
        else:
            print(f"⚠️ File does not exist: {file_path}")
    except Exception as e:
        print(f"❌ Error removing {file_path}: {e}")

def main():
    """Main cleanup function"""
    # Get the project root
    project_root = Path(__file__).parent

    print("\n🗑️ Removing unwanted files...\n")

    # 1. Duplicate files in root directory
    print("\n🗑️ Removing duplicate files...\n")
    duplicate_files = [
        "classification_mcp_server.py",
        "classification_microservice.py",
        "classification_microservice_v2.py",
        "quality_mcp_server.py",
        "api_gateway.py",
        "mcp_contracts.py",
        "mcp_orchestrator.py",
    ]
    
    for file in duplicate_files:
        remove_file(project_root / file)
    
    # 2. Redundant documentation files
    print("\n🗑️ Removing redundant documentation...\n")
    redundant_docs = [
        "ALL_SERVERS_RUNNING.md",
        "DOCUMENT_UPLOAD_FLOW_EXPLAINED.md",
        "FINAL_CLEAN_SUMMARY.md",
        "LOGS_ANALYSIS.md",
        "MCP_CLIENT_SERVER_MAP.md",
        "MICROSERVICES_COMPLETE.md",
        "MICROSERVICES_GUIDE.md",
        "SYSTEM_RUNNING.md",
        "PROJECT_STRUCTURE.md",  # This is replaced by CLEAN_STRUCTURE_GUIDE.md
    ]
    
    for file in redundant_docs:
        remove_file(project_root / file)
    
    # 3. Multiple startup scripts
    print("\n🗑️ Removing redundant startup scripts...\n")
    redundant_scripts = [
        "start_all_services.py",
        "start_complete_system.py",
        "start_frontend.py",
        "start_service.py",
        "start_services_simple.py",
        "setup_mcp_system.py",
        "demo_upload_flow.py",
        "run_microservices.py",
        "frontend_server.py",
    ]
    
    for file in redundant_scripts:
        remove_file(project_root / file)
    
    # 4. Other unwanted files
    print("\n🗑️ Removing other unwanted files...\n")
    other_unwanted = [
        "document_classification.log",  # Log files should not be in version control
        "health_check.py",  # Should be part of src/utils/
    ]
    
    for file in other_unwanted:
        remove_file(project_root / file)
    
    # 5. After cleanup is done, can remove this script and previous cleanup script
    print("\n🧹 Cleanup scripts can also be removed if desired...\n")
    print("- cleanup_project.py")
    print("- remove_unwanted_files.py (this script)")
    
    print("\n✅ Unwanted files cleanup completed!\n")
    print("Your project structure should now be clean and organized.")
    print("If there are any issues, you can manually check the recommended structure in CLEAN_STRUCTURE_GUIDE.md")

if __name__ == "__main__":
    main()

# Document Processing System - Clean Structure Guide

This guide provides recommendations for cleaning up the project structure and removing unwanted files.

## Files to Remove

1. **Duplicate files in root directory that have proper versions in src/**
   - `classification_mcp_server.py` (use the one in src/mcp_servers/)
   - `classification_microservice.py` and `classification_microservice_v2.py` (use src/microservices/classification_service.py)
   - `quality_mcp_server.py` (use the one in src/mcp_servers/)
   - `api_gateway.py` (use the one in src/microservices/)
   - `mcp_contracts.py` (use the one in src/utils/)
   - `mcp_orchestrator.py` (use the one in src/utils/)

2. **Duplicate documentation files**
   - Move essential documentation to docs/ folder
   - Remove redundant files:
     - `ALL_SERVERS_RUNNING.md`
     - `DOCUMENT_UPLOAD_FLOW_EXPLAINED.md` (consolidate with implementation guide)
     - `FINAL_CLEAN_SUMMARY.md`
     - `LOGS_ANALYSIS.md`
     - `mcp_architecture_plan.md` (keep only in docs/)
     - `MCP_CLIENT_SERVER_MAP.md`
     - `mcp_communication_guide.md` (keep only in docs/)
     - `MCP_CONTRACTS_EXPLAINED.md` (keep only in docs/)
     - `MCP_SYSTEM_COMPLETE_GUIDE.md` (keep only in docs/)
     - `MICROSERVICES_COMPLETE.md`
     - `MICROSERVICES_GUIDE.md`
     - `SYSTEM_RUNNING.md`

3. **Test files that are not part of a proper tests/ directory**
   - `test_mcp_server_only.py` (move to tests/)
   - `test_mcp_simple.py` (move to tests/)
   - `test_mcp_simplified.py` (move to tests/)

4. **Multiple startup scripts that do similar things**
   - Keep only `run.py` as the main entry point
   - Remove redundant scripts:
     - `start_all_services.py`
     - `start_complete_system.py`
     - `start_frontend.py`
     - `start_service.py`
     - `start_services_simple.py`
     - `setup_mcp_system.py`
     - `demo_upload_flow.py`

5. **Docker files that should be in docker/ directory**
   - Move to docker/:
     - `docker-compose.yml`
     - `docker-compose.microservices.yml`
     - `Dockerfile.classification`
     - `Dockerfile.gateway`
     - `Dockerfile.quality`

## Recommended Project Structure

```
F2/
├── src/                      # All source code
│   ├── mcp_servers/         # MCP Protocol Servers
│   │   ├── __init__.py
│   │   ├── classification_mcp_server.py
│   │   └── quality_mcp_server.py
│   │
│   ├── microservices/       # REST API Microservices
│   │   ├── __init__.py
│   │   ├── classification_service.py
│   │   └── api_gateway.py
│   │
│   ├── core/                # Core Business Logic
│   │   ├── __init__.py
│   │   ├── document_classifier.py
│   │   └── quality_analyzer.py
│   │
│   ├── utils/               # Shared Utilities
│   │   ├── __init__.py
│   │   ├── mcp_contracts.py
│   │   └── mcp_orchestrator.py
│   │
│   └── __init__.py
│
├── config/                  # Configuration Files
│   ├── classification_config.json
│   └── TresholdConfig.json
│
├── tests/                   # Test files
│   ├── __init__.py
│   ├── test_classification.py
│   └── test_quality.py
│
├── frontend/                # Web Frontend
│   └── index.html
│
├── docs/                    # Documentation
│   ├── SYSTEM_GUIDE.md
│   └── IMPLEMENTATION_GUIDE.md
│
├── docker/                  # Docker configuration
│   ├── docker-compose.yml
│   ├── Dockerfile.classification
│   ├── Dockerfile.gateway
│   └── Dockerfile.quality
│
├── run.py                   # Main project runner
├── setup.py                 # Package setup
└── requirements.txt         # Python dependencies
```

## Implementation Steps

1. Create the required directories:
   ```powershell
   mkdir -p tests docker
   ```

2. Move Docker files:
   ```powershell
   Move-Item -Path docker-compose.yml -Destination docker/
   Move-Item -Path docker-compose.microservices.yml -Destination docker/docker-compose.microservices.yml
   Move-Item -Path Dockerfile.classification -Destination docker/
   Move-Item -Path Dockerfile.gateway -Destination docker/
   Move-Item -Path Dockerfile.quality -Destination docker/
   ```

3. Create test files and move test content:
   ```powershell
   New-Item -Path tests/__init__.py -Type File -Force
   # Create test_classification.py and test_quality.py in tests/
   ```

4. Clean up redundant scripts:
   ```powershell
   # After backing up any needed content, remove redundant files
   Remove-Item start_all_services.py, start_complete_system.py, start_frontend.py, start_service.py, start_services_simple.py
   ```

5. Update run.py to use the new structure
   - Modify paths to use module imports instead of file paths
   - Example: `sys.executable, "-m", "src.microservices.classification_service"` 
   - Instead of: `sys.executable, str(service_path)`

6. Update setup.py and requirements.txt to ensure proper dependencies

7. Consolidate documentation into docs/ folder

## Run Commands

After restructuring, use these commands to run the system:

```powershell
# Start classification service
python run.py service classification

# Start quality service
python run.py service quality

# Start API gateway
python run.py service gateway

# Run tests
python run.py test

# Start the frontend
python run.py frontend

# Start with Docker
python run.py docker
```

## Microservices Integration

The microservices in this structure communicate as follows:

1. API Gateway (`src/microservices/api_gateway.py`) - Port 8000
   - Routes requests to appropriate services
   - Provides unified API interface

2. Classification Service (`src/microservices/classification_service.py`) - Port 8001
   - Handles document classification requests
   - Uses Classification MCP Server for processing

3. Quality Service - Port 8002
   - Analyzes document quality
   - Uses Quality MCP Server for processing

4. MCP Servers (src/mcp_servers/*)
   - Provide JSON-RPC 2.0 interfaces for core functionality
   - Run as subprocess or standalone servers

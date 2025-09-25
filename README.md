# MCP Microservices System

This project implements a document processing system using a microservices architecture with the Model Context Protocol (MCP) for communication between services.

## System Architecture

The system consists of the following microservices:

- **API Gateway** (Port 8000): Entry point for all client requests
- **Classification Service** (Port 8001): Handles document classification
- **Quality Service** (Port 8002): Analyzes document quality
- **Redis**: Used for caching and session management
- **Frontend** (Port 80): Web interface for the system

## Running with Docker

### Prerequisites

- Docker and Docker Compose installed
- Python 3.8 or higher

### Starting the System

You can run the entire system using Docker Compose with a single command:

```bash
python run_docker.py
```

This script will:
1. Check if Docker and Docker Compose are installed
2. Build all service images
3. Start all containers in detached mode
4. Display the status of all services

### Checking Service Health

To verify that all services are running correctly:

```bash
python check_health.py
```

This script will check the health endpoints of all services and report their status.

### Stopping the System

To stop all services:

```bash
python stop_docker.py
```

### Viewing Logs

To view the logs from all services:

```bash
docker-compose logs -f
```

To view logs for a specific service:

```bash
docker-compose logs -f [service-name]
```

Replace `[service-name]` with one of: `api-gateway`, `classification-service`, `quality-service`, `redis`, or `frontend`.

## Manual Operation

If you prefer to run Docker Compose commands directly:

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Rebuild and start services
docker-compose up --build -d
```

## Accessing the Services

- Frontend: http://localhost:80
- API Gateway: http://localhost:8000
- Classification Service: http://localhost:8001
- Quality Service: http://localhost:8002

## Development

For development without Docker, see the [IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) file.

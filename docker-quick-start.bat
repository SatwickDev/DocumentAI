@echo off
echo Quick Start - F2 Document Processing Services in Docker
echo =====================================================

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running or not installed!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo Docker is running. Starting services...
echo.

REM Stop any existing containers
echo Stopping existing containers...
docker-compose -f docker-compose.simple.yml down 2>nul

REM Start services
echo Starting services with docker-compose...
docker-compose -f docker-compose.simple.yml up -d

REM Wait for services
echo.
echo Waiting for services to initialize...
timeout /t 15 /nobreak >nul

REM Check status
echo.
echo Checking service status...
docker-compose -f docker-compose.simple.yml ps

echo.
echo =====================================================
echo Services should be available at:
echo.
echo   API Gateway:               http://localhost:8000
echo   API Gateway Docs:          http://localhost:8000/docs
echo   Classification Service:    http://localhost:8001
echo   Quality Service:           http://localhost:8002
echo   Preprocessing Service:     http://localhost:8003
echo   Entity Extraction Service: http://localhost:8004
echo   Frontend:                  http://localhost:8080
echo.
echo Opening frontend in browser...
timeout /t 3 /nobreak >nul
start http://localhost:8080

echo.
echo Commands:
echo   View logs:    docker-compose -f docker-compose.simple.yml logs -f
echo   Stop all:     docker-compose -f docker-compose.simple.yml down
echo   Restart:      Run this script again
echo.
pause
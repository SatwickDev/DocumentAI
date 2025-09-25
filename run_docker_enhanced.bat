@echo off
echo ========================================
echo Starting Enhanced Docker Services
echo ========================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not installed.
    echo Please install Docker Desktop and enable WSL 2 integration.
    echo.
    echo Download from: https://www.docker.com/products/docker-desktop
    echo.
    pause
    exit /b 1
)

echo Docker is running. Starting services...
echo.

REM Use the enhanced docker-compose file
docker-compose -f docker-compose.enhanced.yml up --build -d

echo.
echo Waiting for services to start (30 seconds)...
timeout /t 30 /nobreak

echo.
echo ========================================
echo Service Status:
echo ========================================
docker-compose -f docker-compose.enhanced.yml ps

echo.
echo ========================================
echo Services are available at:
echo ========================================
echo - API Gateway:          http://localhost:8000
echo - API Documentation:    http://localhost:8000/docs
echo - Classification:       http://localhost:8001/docs
echo - Quality Analysis:     http://localhost:8002/docs
echo - Preprocessing:        http://localhost:8003/docs
echo - Entity Extraction:    http://localhost:8004/docs
echo.
echo To view logs: docker-compose -f docker-compose.enhanced.yml logs -f
echo To stop: docker-compose -f docker-compose.enhanced.yml down
echo.
pause
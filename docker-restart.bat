@echo off
echo Restarting F2 Document Processing Services in Docker...
echo ==============================================

REM Stop and remove existing containers
echo Stopping existing containers...
docker-compose -f docker-compose-full.yml down

REM Remove old images (optional - comment out if you want to keep them)
echo Removing old images...
docker-compose -f docker-compose-full.yml rm -f

REM Build new images
echo Building Docker images...
docker-compose -f docker-compose-full.yml build --no-cache

REM Start services
echo Starting services...
docker-compose -f docker-compose-full.yml up -d

REM Wait for services to be ready
echo Waiting for services to initialize...
timeout /t 10 /nobreak >nul

REM Check service health
echo Checking service health...
docker-compose -f docker-compose-full.yml ps

echo.
echo Service URLs:
echo   API Gateway:               http://localhost:8000
echo   API Gateway Docs:          http://localhost:8000/docs
echo   Classification Service:    http://localhost:8001
echo   Quality Service:           http://localhost:8002
echo   Preprocessing Service:     http://localhost:8003
echo   Entity Extraction Service: http://localhost:8004
echo   Frontend:                  http://localhost:8080
echo.
echo To view logs: docker-compose -f docker-compose-full.yml logs -f
echo To stop services: docker-compose -f docker-compose-full.yml down
echo.
echo Services restarted successfully!
pause
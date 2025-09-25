@echo off
echo Starting Docker services...
docker-compose up --build -d
echo.
echo Services are starting...
timeout /t 10
echo.
echo Checking service status...
docker-compose ps
echo.
echo Services should be available at:
echo - API Gateway: http://localhost:8000
echo - API Docs: http://localhost:8000/docs
echo.
echo To view logs: docker-compose logs -f
echo To stop: docker-compose down
pause
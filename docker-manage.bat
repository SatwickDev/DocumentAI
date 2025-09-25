@echo off
setlocal enabledelayedexpansion

:menu
cls
echo ========================================
echo F2 Document Processing - Docker Manager
echo ========================================
echo.
echo 1. Start all services
echo 2. Stop all services  
echo 3. Restart all services
echo 4. View service logs
echo 5. Check service health
echo 6. Rebuild services (clean build)
echo 7. Remove all containers and images
echo 8. Open Frontend in browser
echo 9. Open API Documentation
echo 0. Exit
echo.
set /p choice=Enter your choice: 

if "%choice%"=="1" goto start
if "%choice%"=="2" goto stop
if "%choice%"=="3" goto restart
if "%choice%"=="4" goto logs
if "%choice%"=="5" goto health
if "%choice%"=="6" goto rebuild
if "%choice%"=="7" goto clean
if "%choice%"=="8" goto frontend
if "%choice%"=="9" goto apidocs
if "%choice%"=="0" goto end

echo Invalid choice. Please try again.
pause
goto menu

:start
echo.
echo Starting all services...
docker-compose -f docker-compose.simple.yml up -d
echo.
echo Services started. Waiting for initialization...
timeout /t 10 /nobreak >nul
docker-compose -f docker-compose.simple.yml ps
pause
goto menu

:stop
echo.
echo Stopping all services...
docker-compose -f docker-compose.simple.yml down
echo Services stopped.
pause
goto menu

:restart
echo.
echo Restarting all services...
docker-compose -f docker-compose.simple.yml restart
echo Services restarted.
pause
goto menu

:logs
echo.
echo Showing logs (Press Ctrl+C to stop)...
docker-compose -f docker-compose.simple.yml logs -f
pause
goto menu

:health
echo.
echo Checking service health...
python docker-health-check.py
pause
goto menu

:rebuild
echo.
echo Rebuilding all services...
docker-compose -f docker-compose.simple.yml down
docker-compose -f docker-compose.simple.yml build --no-cache
docker-compose -f docker-compose.simple.yml up -d
echo.
echo Services rebuilt and started.
pause
goto menu

:clean
echo.
echo WARNING: This will remove all containers and images!
set /p confirm=Are you sure? (y/n): 
if /i "%confirm%"=="y" (
    docker-compose -f docker-compose.simple.yml down
    docker-compose -f docker-compose.simple.yml rm -f
    docker system prune -a -f
    echo All containers and images removed.
) else (
    echo Operation cancelled.
)
pause
goto menu

:frontend
echo.
echo Opening frontend in browser...
start http://localhost:8080
goto menu

:apidocs
echo.
echo Opening API documentation...
start http://localhost:8000/docs
goto menu

:end
echo.
echo Goodbye!
exit /b 0
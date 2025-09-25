@echo off
echo Starting Enhanced Services on Windows...
echo ======================================

cd /d "C:\ilyas\work\python\F2"

REM Start Preprocessing Service
echo.
echo Starting Preprocessing Service (Port 8003)...
start "Preprocessing Service" /D "C:\ilyas\work\python\F2\microservices\preprocessing-service" cmd /c "..\..\venv-3.11\Scripts\python.exe app_simple.py"

REM Wait a bit
timeout /t 3 /nobreak >nul

REM Start Entity Extraction Service
echo Starting Entity Extraction Service (Port 8004)...
start "Entity Extraction Service" /D "C:\ilyas\work\python\F2\microservices\entity-extraction-service" cmd /c "..\..\venv-3.11\Scripts\python.exe app_simple.py"

REM Wait for services to start
echo.
echo Waiting for services to initialize...
timeout /t 5 /nobreak >nul

REM Show status
echo.
echo ======================================
echo Services should now be running at:
echo.
echo Preprocessing Service:     http://localhost:8003
echo   Health Check:           http://localhost:8003/health
echo.
echo Entity Extraction Service: http://localhost:8004
echo   Health Check:           http://localhost:8004/health
echo.
echo The frontend should now detect these services!
echo ======================================
echo.
pause
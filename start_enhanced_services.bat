@echo off
echo Starting Enhanced Services (Preprocessing & Entity Extraction)...
echo ==========================================================

cd /d C:\ilyas\work\python\F2

REM Kill any existing processes on ports 8003 and 8004
echo Stopping any existing services...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8003" ^| find "LISTENING"') do taskkill /F /PID %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8004" ^| find "LISTENING"') do taskkill /F /PID %%a 2>nul

timeout /t 2 /nobreak >nul

REM Start Preprocessing Service (Simple Version)
echo.
echo Starting Preprocessing Service (port 8003)...
start "Preprocessing Service" cmd /k "cd microservices\preprocessing-service && ..\..\venv-3.11\Scripts\python.exe app_simple.py"

timeout /t 3 /nobreak >nul

REM Start Entity Extraction Service (Simple Version)
echo Starting Entity Extraction Service (port 8004)...
start "Entity Extraction Service" cmd /k "cd microservices\entity-extraction-service && ..\..\venv-3.11\Scripts\python.exe app_simple.py"

timeout /t 3 /nobreak >nul

echo.
echo Services should now be running!
echo.
echo Testing services...
timeout /t 3 /nobreak >nul

REM Test the services
echo.
venv-3.11\Scripts\python.exe -c "import requests; print('Preprocessing Service:', 'OK' if requests.get('http://localhost:8003/health').status_code == 200 else 'FAILED')"
venv-3.11\Scripts\python.exe -c "import requests; print('Entity Extraction Service:', 'OK' if requests.get('http://localhost:8004/health').status_code == 200 else 'FAILED')"

echo.
echo Service URLs:
echo   Preprocessing Service:     http://localhost:8003
echo   Entity Extraction Service: http://localhost:8004
echo.
echo You can now refresh your browser to use these services!
echo.
pause
@echo off
echo Starting missing services...

cd /d C:\ilyas\work\python\F2

REM Check if API Gateway is running
netstat -an | find "8000" >nul
if errorlevel 1 (
    echo Starting API Gateway...
    start "API Gateway" cmd /k "cd microservices\api-gateway && ..\..\venv-3.11\Scripts\python.exe app.py"
) else (
    echo API Gateway already running on port 8000
)

timeout /t 3 /nobreak >nul

REM Check if Preprocessing Service is running
netstat -an | find "8003" >nul
if errorlevel 1 (
    echo Starting Preprocessing Service...
    start "Preprocessing Service" cmd /k "cd microservices\preprocessing-service && ..\..\venv-3.11\Scripts\python.exe app.py"
) else (
    echo Preprocessing Service already running on port 8003
)

timeout /t 3 /nobreak >nul

REM Check if Entity Extraction Service is running
netstat -an | find "8004" >nul
if errorlevel 1 (
    echo Starting Entity Extraction Service...
    start "Entity Extraction Service" cmd /k "cd microservices\entity-extraction-service && ..\..\venv-3.11\Scripts\python.exe app.py"
) else (
    echo Entity Extraction Service already running on port 8004
)

timeout /t 3 /nobreak >nul

REM Check if Frontend is running
netstat -an | find "8080" >nul
if errorlevel 1 (
    echo Starting Frontend...
    start "Frontend" cmd /k "venv-3.11\Scripts\python.exe serve_frontend.py"
) else (
    echo Frontend already running on port 8080
)

echo.
echo All services should now be running!
echo.
echo Service URLs:
echo   API Gateway:               http://localhost:8000
echo   Classification Service:    http://localhost:8001
echo   Quality Service:           http://localhost:8002
echo   Preprocessing Service:     http://localhost:8003
echo   Entity Extraction Service: http://localhost:8004
echo   Frontend:                  http://localhost:8080
echo.
pause
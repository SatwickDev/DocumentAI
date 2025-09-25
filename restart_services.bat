@echo off
echo Stopping existing Python services...
taskkill /F /IM python.exe 2>nul

echo.
echo Starting F2 Document Processing Services...
echo ==========================================

cd /d C:\ilyas\work\python\F2

REM Start API Gateway
echo Starting API Gateway (port 8000)...
start "API Gateway" cmd /k "cd microservices\api-gateway && ..\..\venv-3.11\Scripts\python.exe app.py"
timeout /t 5 /nobreak >nul

REM Start Classification Service
echo Starting Classification Service (port 8001)...
start "Classification Service" cmd /k "cd microservices\classification-service && ..\..\venv-3.11\Scripts\python.exe app.py"
timeout /t 3 /nobreak >nul

REM Start Quality Service  
echo Starting Quality Service (port 8002)...
start "Quality Service" cmd /k "cd microservices\quality-service && ..\..\venv-3.11\Scripts\python.exe app.py"
timeout /t 3 /nobreak >nul

REM Start Preprocessing Service
echo Starting Preprocessing Service (port 8003)...
start "Preprocessing Service" cmd /k "cd microservices\preprocessing-service && ..\..\venv-3.11\Scripts\python.exe app.py"
timeout /t 3 /nobreak >nul

REM Start Entity Extraction Service
echo Starting Entity Extraction Service (port 8004)...
start "Entity Extraction Service" cmd /k "cd microservices\entity-extraction-service && ..\..\venv-3.11\Scripts\python.exe app.py"
timeout /t 3 /nobreak >nul

REM Start Frontend
echo Starting Frontend (port 8080)...
start "Frontend" cmd /k "venv-3.11\Scripts\python.exe serve_frontend.py"
timeout /t 3 /nobreak >nul

echo.
echo ==========================================
echo All services started!
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
echo Opening frontend in browser...
timeout /t 5 /nobreak >nul
start http://localhost:8080

echo.
echo Services are running. Press any key to stop all services...
pause >nul

echo.
echo Stopping all services...
taskkill /F /IM python.exe
echo All services stopped.
pause
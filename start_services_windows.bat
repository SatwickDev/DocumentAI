@echo off
echo Starting F2 Document Processing Services...
echo ==========================================

REM Start API Gateway
start "API Gateway" cmd /k "cd microservices\api-gateway && ..\..\venv-3.11\Scripts\python.exe app.py"
timeout /t 5 /nobreak

REM Start Classification Service
start "Classification Service" cmd /k "cd microservices\classification-service && ..\..\venv-3.11\Scripts\python.exe app.py"
timeout /t 5 /nobreak

REM Start Quality Service  
start "Quality Service" cmd /k "cd microservices\quality-service && ..\..\venv-3.11\Scripts\python.exe app.py"
timeout /t 5 /nobreak

REM Start Preprocessing Service
start "Preprocessing Service" cmd /k "cd microservices\preprocessing-service && ..\..\venv-3.11\Scripts\python.exe app.py"
timeout /t 5 /nobreak

REM Start Entity Extraction Service
start "Entity Extraction Service" cmd /k "cd microservices\entity-extraction-service && ..\..\venv-3.11\Scripts\python.exe app.py"
timeout /t 5 /nobreak

REM Start Frontend
start "Frontend" cmd /k "venv-3.11\Scripts\python.exe serve_frontend.py"
timeout /t 5 /nobreak

echo ==========================================
echo All services started!
echo.
echo Service URLs:
echo API Gateway: http://localhost:8000
echo Classification Service: http://localhost:8001  
echo Quality Service: http://localhost:8002
echo Preprocessing Service: http://localhost:8003
echo Entity Extraction Service: http://localhost:8004
echo Frontend: http://localhost:8080
echo.
echo Opening frontend in browser...
start http://localhost:8080
echo.
echo Press any key to stop all services...
pause

REM Kill all python processes (be careful with this)
taskkill /F /IM python.exe
echo Services stopped.
pause
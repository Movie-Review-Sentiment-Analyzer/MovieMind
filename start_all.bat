@echo off
echo ===================================================
echo     Starting MovieMind Full-Stack NLP System
echo ===================================================
echo.

echo [1/3] Starting Python NLP Sentiment Engine on port 8000...
start "MovieMind NLP Engine" cmd /k "python src/sentiment_api.py"

timeout /t 2 >nul

echo [2/3] Starting Node.js Backend Server on port 5000...
start "MovieMind Node Backend" cmd /k "cd backend && node server.js"

timeout /t 2 >nul

echo [3/3] Starting React Frontend on port 5173...
start "MovieMind React Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ===================================================
echo All services launched!
echo - Frontend: http://localhost:5173
echo - Live NLP Tester: http://localhost:5173/analyzer
echo - Backend API: http://localhost:5000
echo - Python NLP Service: http://127.0.0.1:8000
echo ===================================================
timeout /t 4 >nul
start http://localhost:5173

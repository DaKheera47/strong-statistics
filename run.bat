@echo off
REM Lifting Pipeline launcher (development convenience)
REM Usage: run.bat [port]
REM Example: run.bat 8001

setlocal ENABLEDELAYEDEXPANSION
set BASE_DIR=%~dp0
cd /d %BASE_DIR%

REM Ensure virtual environment exists
if not exist "%BASE_DIR%\.venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment .venv ...
  py -m venv .venv || goto :fail
)

REM Activate venv
call "%BASE_DIR%\.venv\Scripts\activate.bat" || goto :fail

REM Install deps if fastapi missing
python -c "import fastapi" 2>NUL 1>NUL
if errorlevel 1 (
  echo [INFO] Installing requirements...
  pip install -q -r requirements.txt || goto :fail
)

set HOST=127.0.0.1
set PORT=8000
if not "%1"=="" set PORT=%1

if exist .env (
  echo [INFO] Using .env file.
) else (
  echo [WARN] .env not found. Copy .env.example to .env and set INGEST_TOKEN.
)

echo [START] Lifting Pipeline at http://%HOST%:%PORT% (Ctrl+C to stop)
uvicorn app.main:app --host %HOST% --port %PORT% --reload
set CODE=%ERRORLEVEL%

echo [EXIT] Uvicorn exited with code %CODE%
endlocal & exit /b %CODE%

:fail
echo [ERROR] Startup failed.
endlocal & exit /b 1

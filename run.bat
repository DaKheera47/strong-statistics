@echo off
REM strong-statistics launcher (development convenience)
REM Usage: run.bat [host] [port]
REM Examples:
REM   run.bat            -> 0.0.0.0:8069 (LAN accessible)
REM   run.bat 127.0.0.1  -> 127.0.0.1:8069 (loopback only)
REM   run.bat 0.0.0.0 9000 -> 0.0.0.0:9000

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

REM Determine host / port (allow env override, then args)
set HOST=%HOST%
if "%HOST%"=="" set HOST=0.0.0.0
set PORT=8069
if not "%1"=="" (
  REM If first arg contains a dot assume it's a host, else treat as port
  echo %1 | findstr /R /C:"[0-9][0-9]*\.[0-9]" >NUL && (set HOST=%1) || (set PORT=%1)
)
if not "%2"=="" set PORT=%2

if exist .env (
  echo [INFO] Using .env file.
) else (
  echo [WARN] .env not found. Copy .env.example to .env and set INGEST_TOKEN.
)

echo [START] strong-statistics binding %HOST%:%PORT% (Ctrl+C to stop)
uvicorn app.main:app --host %HOST% --port %PORT% --reload
set CODE=%ERRORLEVEL%

echo [EXIT] Uvicorn exited with code %CODE%
endlocal & exit /b %CODE%

:fail
echo [ERROR] Startup failed.
endlocal & exit /b 1

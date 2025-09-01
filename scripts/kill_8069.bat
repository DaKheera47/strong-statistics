@echo off
REM Kill all processes using TCP/UDP port 8069
set PORT=8069

echo Searching for processes using port %PORT% ...
netstat -ano | findstr :%PORT% >nul
if errorlevel 1 (
  echo No processes found using port %PORT%.
  goto :eof
)

setlocal enabledelayedexpansion
for /f "tokens=5" %%P in ('netstat -ano ^| findstr :%PORT%') do (
  if not defined SEEN_%%P (
    set SEEN_%%P=1
    echo Killing PID %%P
    taskkill /F /PID %%P >nul 2>&1
    if errorlevel 1 (
      echo   Failed to kill PID %%P (may require Administrator privileges)
    ) else (
      echo   PID %%P terminated.
    )
  )
)
endlocal

echo Done.

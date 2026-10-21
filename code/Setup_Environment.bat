@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Setup_Environment.ps1"
if errorlevel 1 (
  echo.
  echo Environment setup failed. See the message above.
  pause
  exit /b 1
)
echo.
echo Environment setup completed successfully.
pause

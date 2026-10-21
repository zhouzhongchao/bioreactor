@echo off
setlocal
cd /d "%~dp0"
set "APP_PYTHON=%~dp0..\data\.python310\python.exe"
if defined TRO_DATA_ROOT set "APP_PYTHON=%TRO_DATA_ROOT%\.python310\python.exe"
if not exist "%APP_PYTHON%" (
  echo Environment is not installed. Run Setup_Environment.bat first.
  pause
  exit /b 1
)
set PYTHONDONTWRITEBYTECODE=1
"%APP_PYTHON%" "%~dp0portable_check.py"
pause

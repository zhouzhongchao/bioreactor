@echo off
setlocal
cd /d "%~dp0"
set "APP_PYTHON=%~dp0..\data\.python310\python.exe"
if defined TRO_DATA_ROOT set "APP_PYTHON=%TRO_DATA_ROOT%\.python310\python.exe"
if not exist "%APP_PYTHON%" (
  echo First start: preparing the local Python environment...
  call "%~dp0Setup_Environment.bat"
  if errorlevel 1 exit /b 1
)
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
"%APP_PYTHON%" "%~dp0qt_gui.py"
if errorlevel 1 (
  echo.
  echo The application exited with an error.
  pause
)

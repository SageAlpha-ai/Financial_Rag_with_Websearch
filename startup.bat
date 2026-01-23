@echo off
REM Azure App Service startup script for Windows
REM This script is used by Azure App Service to start the FastAPI application

REM Azure App Service automatically sets the PORT environment variable
REM If PORT is not set, default to 8000 (for local development)
if "%PORT%"=="" set PORT=8000

REM Start uvicorn with the app
REM Note: Azure App Service requires binding to 0.0.0.0 and using the PORT variable
uvicorn api:app --host 0.0.0.0 --port %PORT% --workers 1

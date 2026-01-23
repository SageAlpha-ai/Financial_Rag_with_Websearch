#!/bin/bash
# Azure App Service Startup Script

# Install dependencies if needed
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Start the application
# Azure App Service automatically sets PORT environment variable
exec gunicorn api:app --bind 0.0.0.0:${PORT:-8000} --workers 2 --timeout 300 --worker-class uvicorn.workers.UvicornWorker --access-logfile - --error-logfile -

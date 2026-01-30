"""
Gunicorn configuration for Azure App Service
"""

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker processes
# Fixed to 1 worker for graceful shutdown and reduced initialization overhead
# Override with WEB_CONCURRENCY env var if needed (e.g., WEB_CONCURRENCY=2)
workers = int(os.getenv('WEB_CONCURRENCY', '1'))
worker_class = 'uvicorn.workers.UvicornWorker'
worker_connections = 1000

# Timeouts for graceful shutdown
timeout = 180  # Worker timeout (seconds)
graceful_timeout = 180  # Graceful shutdown timeout (seconds)
keepalive = 5

# Worker recycling to prevent memory leaks
max_requests = 1000  # Restart worker after N requests
max_requests_jitter = 50  # Random jitter to prevent all workers restarting at once

# Logging
accesslog = '-'
errorlog = '-'
loglevel = os.getenv('LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'financial-rag-api'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
keyfile = None
certfile = None

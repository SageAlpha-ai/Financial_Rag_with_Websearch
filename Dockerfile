# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port 8000 (Azure App Service will map this via WEBSITES_PORT)
EXPOSE 8000

# Start application using gunicorn with uvicorn workers for FastAPI
# IMPORTANT: Environment variables must be provided via --env-file or -e flags
# Example: docker run --env-file .env <image>
# DO NOT bake secrets into the image
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]

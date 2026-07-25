# Use Python 3.11 slim image for a smaller footprint and updated SQLite
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for:
# - PyMuPDF (libsqlite3-dev, build-essential)
# We omit Playwright dependencies to save space and RAM for e2-micro
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Create necessary directories for state and data
RUN mkdir -p /data

# Copy the rest of the application
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Environment variables for production storage

ENV REGISTRY_DB_PATH=/data/registry.db
ENV PYTHONIOENCODING=utf-8

# Start uvicorn in the foreground
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

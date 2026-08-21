# Use official Python runtime as base image
FROM python:3.10-slim

# Install system dependencies required by OpenCV and Python build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (Render automatically sets $PORT)
EXPOSE 5000

# Start command using Gunicorn bound to 0.0.0.0:$PORT
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} app:app

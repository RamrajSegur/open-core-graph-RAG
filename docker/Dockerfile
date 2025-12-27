# Multi-stage build for optimized image size
FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed

# Expose port for API (future use)
EXPOSE 8000

# Default command
CMD ["python", "-m", "ipython"]

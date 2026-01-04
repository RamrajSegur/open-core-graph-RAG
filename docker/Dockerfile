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

# Download SpaCy language model for NER (with explicit URL for compatibility)
RUN python -m spacy download en_core_web_sm --user 2>/dev/null || \
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Copy project source code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed

# Expose port for API (future use)
EXPOSE 8000

# Default command
CMD ["python", "-m", "ipython"]

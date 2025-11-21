# CKD Stage Predictor - Docker Configuration
FROM python:3.11-slim

# Set metadata
LABEL maintainer="e-RickReyJim"
LABEL description="CKD Stage Predictor - 5-Stage Classification with ML"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY LICENSE .
COPY README.md .

# Create non-root user for security
RUN useradd -m -u 1000 ckduser && \
    chown -R ckduser:ckduser /app

# Switch to non-root user
USER ckduser

# Expose Gradio default port
EXPOSE 7870

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7870')" || exit 1

# Set entrypoint
ENTRYPOINT ["python", "src/app.py"]

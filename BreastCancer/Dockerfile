FROM python:3.10-slim

ENV TRANSFORMERS_CACHE=/tmp/.cache
ENV HF_HOME=/tmp/.cache
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p /app/models

# Set proper permissions
RUN chmod -R 755 /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
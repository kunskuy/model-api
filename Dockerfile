# Base image Python
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt & install dependencies Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code to container
COPY src/ .

# Create directory
RUN mkdir -p /app/uploads /app/model

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Ekspose port 8080
EXPOSE 8080

# Run app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
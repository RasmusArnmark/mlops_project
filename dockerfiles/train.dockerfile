# Use a lightweight Python base image
FROM python:3.11-slim AS base

# Install only the necessary system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file to cache dependencies
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ /app/src/
COPY models/ /app/models/

# Ensure data directory exists (Optional)
RUN mkdir -p /app/data/

# Set environment variables (example for Google Cloud credentials)
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/src/service-account.json

# Default command to execute the training script
ENTRYPOINT ["python", "src/train.py"]

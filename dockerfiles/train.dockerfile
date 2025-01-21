# Base image with Python 3.11
FROM python:3.11-slim AS base

# Install necessary system dependencies for building Python packages
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to cache pip installations
COPY requirements.txt /app/

# Install Python dependencies (including DVC with GCS support)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Set environment variables for GCS and W&B
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json
ENV WANDB_API_KEY=${WANDB_API_KEY}

# Default command to execute the training script
ENTRYPOINT ["python", "src/train.py"]

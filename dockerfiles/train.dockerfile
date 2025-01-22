# Use a lightweight Python base image
FROM python:3.11-slim AS base

# Install only the necessary system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the entire application directory, including .gitignored files excluded
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command to execute the training script
ENTRYPOINT ["python", "src/train.py"]

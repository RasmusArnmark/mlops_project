# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies for gsutil
RUN apt-get update && apt-get install -y \
    curl \
    gnupg && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y google-cloud-sdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY src/ src/
COPY data/ data/

# Create the output directories for data
RUN mkdir -p /data/raw /data/processed

# Run the data preprocessing script
CMD ["bash", "-c", "python src/data.py && cp -r /data /output"]

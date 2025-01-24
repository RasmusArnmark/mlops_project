# Food Image Classification Project

## Overview

This project demonstrates an end-to-end MLOps pipeline for training and evaluating a Convolutional Neural Network (CNN) model to classify food images. The pipeline follows best practices for reproducibility and scalability and utilizes the Food Image Classification Dataset from Kaggle. Key components include data preprocessing, training, and testing.

## Dataset

**Food Image Classification Dataset**  
- **Source**: [Kaggle](https://www.kaggle.com/code/gauravduttakiit/class-dataset-food-image-classification/data)  
- **Description**: Contains images of various food items categorized into distinct classes such as burgers, pizzas, samosas, fries, and more.  
- **Structure**: Organized into subdirectories for each class.

## Getting Started

### 1. Prerequisites

Ensure you have the following installed:
- Python 3.11
- pip

Install dependencies:

pip install -r requirements.txt

## Objectives
1. Build a robust classification model for food classification
2. Implement ML Ops practices for reproducibility, scalability, and efficient deployment.

## Tools and Frameworks
- **Model**: Convolutional Neural Network
- **Programming Framework**: PyTorch for model fine-tuning.
- **Data Handling**: Pandas and NumPy for preprocessing.
- **Version Control**: Git and DVC.
- **Experiment Tracking**: Weights & Biases (W&B).
- **Deployment**: Docker for containerization.


### 2. How to run
## Setup Instructions

This section will guide you through setting up the project both locally and using Google Cloud Platform (GCP). Follow these steps to get started.

---

### Local Setup

1. **Create a Conda Environment:**

   Ensure you have Conda installed on your system. Create and activate a new environment:

   ```bash
   conda create --name mlops_env python=3.11 -y
   conda activate mlops_env
   ```

2. **Install Dependencies:**

   Use the `requirements.txt` file to install project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data:**

   Run the data preparation script to download and preprocess the data:

   ```bash
   docker build -f dockerfiles/data.dockerfile -t data-processor .
   ```

   Then run, this will download and preprocess the data, and mount it to an output directory. We couldnt mount it to the data folder for various reasons.

   ```bash
   docker run --rm  -v $(pwd)/output_dir:/app/output data-processor
   ```

   Lastly run 

    ```bash
   mv output_dir/data/* data/ && rm -rf output_dir
   ```
   
4. **Build and Run the Training Docker Image:**

   Build the Docker image for training:

   ```bash
   docker build -t train-image:latest -f dockerfiles/train.dockerfile .
   ```

   Run the training container and set your Weights & Biases (W&B) API key:

   ```bash
   docker run -e WANDB_API_KEY=<your-wandb-api-key> -v $(pwd)/models:/app/models train-image:latest
   ```

   Replace `<your-wandb-api-key>` with your actual W&B API key.

5. **Build and Run the API Docker Image:**

   Build the Docker image for the API:

   ```bash
   docker build -t api-image:latest -f dockerfiles/api.dockerfile .
   ```

   Run the API container:

   ```bash
   docker run -p 8000:8000 api-image:latest
   ```

   Access the API at [http://localhost:8000/docs](http://localhost:8000/docs) to test predictions.

---

### Optional GCP Setup

To leverage GCP for model storage, automated Docker builds, and deployment, follow these steps:

#### 1. **Set Up a GCS Bucket:**

   Create a bucket to store models and data:

   ```bash
   gcloud storage buckets create gs://<your-gcs-bucket> --location=<your-region>
   ```

   Update the `.env` file to include the GCS bucket name:

   ```env
   GCS_BUCKET=<your-gcs-bucket>
   ```

   The trained model will automatically be uploaded to the bucket during the training process.

#### 2. **Set Up GitHub Actions for Automated Builds:**

   Configure GitHub Actions to build and push your Docker images to GCP Artifact Registry. Add your GCP credentials and W&B API key as secrets in your GitHub repository.

   Once configured, any push to the `main` branch will trigger GitHub Actions to build and push the Docker images for training and API.

#### 3. **Run Training and Deployment on GCP:**

   - **Training:** Use Vertex AI to run the training job. Example configuration for `vertex_ai_job.yaml` is included in the repository.
   - **Deployment:** Use Cloud Run to deploy the API. The following command deploys the Gradio app:

     ```bash
     gcloud run deploy food-classification-api \
       --image=europe-west1-docker.pkg.dev/<your-project-id>/food-class/api-image:latest \
       --region=europe-west1 \
       --allow-unauthenticated
     ```

By following these steps, you can set up the project locally and extend it to utilize GCP for scalable training, deployment, and monitoring.

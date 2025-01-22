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
- Python 3.8+
- pip
- PyTorch
- torchvision
- DVC
- scikit-learn

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

Create bucket in gcs

Download credential-key for serviceaccount with access to the bucket

Run 
python src/data.py

Build docker image with 
docker build -t food-trainer:latest -f dockerfiles/train.dockerfile .

Run training with docker, remember
docker run --rm \
    -e GOOGLE_APPLICATION_CREDENTIALS=/app/src/credentials.json \
    -v <path-to-your-credentials-file>:/app/src/credentials.json \
    -e WANDB_API_KEY=<your-wandb-api-key> \
    food-trainer:latest

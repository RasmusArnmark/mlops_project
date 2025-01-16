# Food Image Classification Project

## Overview

This project demonstrates an end-to-end MLOps pipeline for training and evaluating a Convolutional Neural Network (CNN) model to classify food images. The pipeline follows best practices for reproducibility and scalability and utilizes the Food Image Classification Dataset from Kaggle. Key components include data preprocessing, training, and testing.

### Project structure

mlops_project/
├── data/               # Dataset directory
│   ├── raw/           # Raw Kaggle dataset
│   └── processed/     # Preprocessed dataset (train/val/test)
├── models/             # Trained model files
├── src/                # Source code
│   ├── data.py         # Data preprocessing script
│   ├── train.py        # Model training script
│   ├── test.py         # Model testing script
│   └── model.py        # CNN model definition
├── dvc.yaml            # DVC pipeline definition
├── params.yaml         # Hyperparameters
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
├── .gitignore          # Files to ignore in Git
└── README.md           # Project documentation

## Dataset

**Food Image Classification Dataset**  
- **Source**: [Kaggle](https://www.kaggle.com)  
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

import os
import numpy as np
import pandas as pd
from PIL import Image
from evidently.metrics import DataDriftTable
from evidently.report import Report

# Define the raw data directory
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")

def load_images_from_directory(directory):
    """Load all images from subdirectories of the given directory."""
    images = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):  # Check if it's a directory
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                try:
                    # Open and convert image to grayscale (if needed)
                    img = Image.open(img_path).convert("L")
                    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
                    images.append(img_array)
                    labels.append(label)  # Use folder name as the label
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return images, labels

# Load images and labels
images, labels = load_images_from_directory(RAW_DATA_DIR)

def extract_features(images):
    """Extract basic image features from a set of images."""
    features = []
    for img in images:
        avg_brightness = np.mean(img)
        contrast = np.std(img)
        sharpness = np.mean(np.abs(np.gradient(img)))
        features.append([avg_brightness, contrast, sharpness])
    return np.array(features)

# Extract features from the loaded images
features = extract_features(images)

# Create a DataFrame
feature_columns = ["Average Brightness", "Contrast", "Sharpness"]
feature_df = pd.DataFrame(features, columns=feature_columns)
feature_df["Dataset"] = labels  # Add labels as the "Dataset" column

# Separate reference and current data
reference_data = feature_df[feature_df["Dataset"] == "reference"].drop(columns=["Dataset"])



current_data = feature_df[feature_df["Dataset"] == "FashionMNIST"].drop(columns=["Dataset"])

# Generate a data drift report
report = Report(metrics=[DataDriftTable()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("data_drift.html")

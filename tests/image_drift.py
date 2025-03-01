import os
import numpy as np
import pandas as pd
from PIL import Image
from evidently.metrics import DataDriftTable
from evidently.report import Report
import json
from google.cloud import storage
import glob
import torch

# Define directories
PROCESSED_DATA_DIR = "data/processed"
NEW_DATA_DIR = "data/new_data"

# Ensure necessary directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(NEW_DATA_DIR, exist_ok=True)

def extract_features(images):
    """
    Extract basic image features from a set of images.
    Features: Average Brightness, Contrast, and Sharpness.
    """
    features = []
    for img in images:
        avg_brightness = np.mean(img)
        contrast = np.std(img)
        sharpness = np.mean(np.abs(np.gradient(img)))
        features.append([avg_brightness, contrast, sharpness])
    return np.array(features)

def load_images_from_directory(directory):
    """
    Load all images from a directory, extracting labels from filenames.
    
    Args:
        directory (str): Path to the directory containing images.

    Returns:
        list: A list of loaded image arrays.
        list: A list of corresponding labels extracted from filenames.
    """
    images, labels = [], []

    # Iterate through all files in the directory
    for img_file in os.listdir(directory):
        img_path = os.path.join(directory, img_file)
        try:
            # Ensure it's a valid image file with the expected format
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')) and '_' in img_file:
                # Extract label from the filename (before the first underscore)
                label = img_file.split('_')[0]
                
                # Load the image
                img = Image.open(img_path)
                
                # Convert the image to a NumPy array
                img_array = np.array(img)
                
                # Append the image and its label
                images.append(img_array)
                labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    return images, labels


def download_images_from_bucket(bucket_name="foodclassrae", local_dir="data"):
    """
    Download JSON files from a GCS bucket, unflatten the image data, 
    and save it as JPG files in the specified directory.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix="new_data/")

    for blob in blobs:
        if blob.name.endswith(".json"):
            # Download each JSON file
            json_filename = os.path.join(local_dir, blob.name)
            os.makedirs(os.path.dirname(json_filename), exist_ok=True)
            blob.download_to_filename(json_filename)
            print(f"Downloaded {blob.name} to {json_filename}")

            # Load the JSON data
            with open(json_filename, "r") as json_file:
                json_data = json.load(json_file)


            if 'image' in json_data:
                image_data = np.array(json_data['image'])
                image_data = (image_data) * 255
                image_data = torch.from_numpy(image_data)
                image_data = image_data.permute(1,2,0).numpy()
                # Convert numpy array to an image (RGB)
                image_data = image_data.astype(np.uint8)
                image = Image.fromarray(image_data, mode="RGB")
                # Prepare the filename
                image.save(f'data/new_data/{json_data['predicted']}_{json_data['timestamp']}.jpg')
            else:   
                print(f"No image data found in {blob.name}")


def load_images_from_processed(directory):
    """
    Load all images from subdirectories of the given directory.
    Each subdirectory represents a class label.
    """
    images, labels = [], []

    # Iterate through the 3 folders in the 'processed' directory
    for main_folder in os.listdir(directory):
        main_folder_path = os.path.join(directory, main_folder)
        
        if os.path.isdir(main_folder_path):  # Check if it's a directory
            # Iterate through the 24 subfolders inside each main folder
            for label in os.listdir(main_folder_path):
                label_dir = os.path.join(main_folder_path, label)
                
                if os.path.isdir(label_dir):  # Ensure it's a directory
                    # Iterate through all images in the subfolder
                    for img_file in os.listdir(label_dir):
                        img_path = os.path.join(label_dir, img_file)
                        
                        try:
                            img = Image.open(img_path) 
                            img_array = np.array(img) 
                            images.append(img_array)
                            labels.append(label)  # Use folder name as the label
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
    return images, labels

if __name__=="__main__":
    print('Begin old features')
    old_images, old_labels = load_images_from_processed("data/processed")
    old_features = extract_features(old_images)

    print('Downloading from bucket')
    download_images_from_bucket()

    new_images, new_label = load_images_from_directory("data/new_data")
    new_feature = extract_features(new_images)


    # Convert old features and labels to a DataFrame
    old_feature_df = pd.DataFrame(
        old_features, 
        columns=["Avg_Brightness", "Contrast", "Sharpness"]  # Replace with your feature names
    )
    old_feature_df["Label"] = old_labels
    old_feature_df["Dataset"] = "Old"

    # Convert new features and labels to a DataFrame
    new_feature_df = pd.DataFrame(
        new_feature, 
        columns=["Avg_Brightness", "Contrast", "Sharpness"]  # Replace with your feature names
    )
    new_feature_df["Label"] = new_label
    new_feature_df["Dataset"] = "New"


    # Separate reference and current data
    print("Generating data drift report...")
    reference_data = old_feature_df.drop(columns=["Dataset"])
    current_data = new_feature_df.drop(columns=["Dataset"])

    # Generate a data drift report
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("data_drift.html")
    print("Data drift report saved.")

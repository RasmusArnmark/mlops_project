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
    Load all images from subdirectories of the given directory.
    Each subdirectory represents a class label.
    """
    images, labels = [], []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):  # Ensure it’s a directory
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                try:
                    img = Image.open(img_path)  # Convert to grayscale
                    img_array = np.array(img)  # Normalize pixel values to [0, 1]
                    images.append(img_array)
                    labels.append(label)  # Use folder name as the label
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return images, labels

def unflatten_data(flat_data):
    """
    Convert a flattened array back into a 2D or 3D image array.
    Assuming the data is a grayscale image represented as a 1D array.
    This function assumes that you know the width and height of the image.
    """

    # Reshape the flattened data into a 2D image array (you may need to adjust this)
    image_data = np.array(flat_data).reshape((3, 128, 128))
    return image_data

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

            # Unflatten the image data
            # Assuming the image data is stored under the key 'image' and is flattened
            if 'image' in json_data:
                image_data = np.array(json_data['image'])

                #flattened_image = json_data['image']
                #image_data = unflatten_data(flattened_image)

                # Reshape to 3x128x128 if needed
                # image_data = image_data.reshape(3, 128, 128)

                # Convert the image data to RGB format
                # Image data is assumed to be normalized between -1 and 1, so scale it to 0-255 range
                image_data = ((image_data + 1) * 127.5)
                #image_data = image_data.reshape(128,128,3)
                image_data = torch.from_numpy(image_data)
                image_data = image_data.permute(1,2,0).numpy
                # Merge the 3 color channels into an RGB image (height x width x 3)
                #image_data = np.moveaxis(image_data, 0, -1)  # Move the channels to the last dimension
                #image_data = image_data.numpy
                # Convert numpy array to an image (RGB)
                image_data = np.array(image_data)
                image = Image.fromarray(image_data.astype(np.uint8), mode="RGB")
                # Prepare the filename
                #image_filename = json_filename.replace(".json", ".jpg")
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
                            img = Image.open(img_path)  # Convert to grayscale
                            img_array = np.array(img)   # Normalize pixel values to [0, 1]
                            images.append(img_array)
                            labels.append(label)  # Use folder name as the label
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
    return images, labels


print('Begin old features')
#old_images, old_labels = load_images_from_processed("data/processed")
#old_features = extract_features(old_images)


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

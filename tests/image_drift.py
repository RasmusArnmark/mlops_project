import os
import numpy as np
import pandas as pd
from PIL import Image
from evidently.metrics import DataDriftTable
from evidently.report import Report
import json
from google.cloud import storage
import glob

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
                    img = Image.open(img_path).convert("L")  # Convert to grayscale
                    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
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

def download_images_from_bucket(bucket_name="foodclassrae", local_dir="data/new_data"):
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
                flattened_image = json_data['image']
                image_data = unflatten_data(flattened_image)
                
                # Convert numpy array to an image (assuming grayscale)
                image = Image.fromarray(image_data.transpose(1,2,0).astype(np.uint8))  # Convert to uint8 for image saving
                image_filename = json_filename.replace(".json", ".jpg")
                
                # Save the image as a .jpg file
                image.save(image_filename)
                print(f"Saved image to {image_filename}")
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
                            img = Image.open(img_path).convert("L")  # Convert to grayscale
                            img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
                            images.append(img_array)
                            labels.append(label)  # Use folder name as the label
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
    return images, labels


print('Begin old features')
old_images, old_labels = load_images_from_processed("data/processed")
old_features = extract_features(old_images)


print('Downloading from bucket')
download_images_from_bucket()

new_images, new_label = load_images_from_directory("data/new_data")
new_feature = extract_features(new_images)

print("hej")


# Process new data
print("Processing new data...")
download_images_from_bucket()  # Download files to NEW_DATA_DIR
new_data_df = load_json_data_to_dataframe()

# Ensure new data includes features
if "Average Brightness" not in new_data_df.columns:
    print("New data does not include features. Processing...")
    new_images, new_labels = load_images_from_directory(NEW_DATA_DIR)
    new_features = extract_features(new_images)

    # Add features to new data
    feature_columns = ["Average Brightness", "Contrast", "Sharpness"]
    new_feature_df = pd.DataFrame(new_features, columns=feature_columns)
    new_feature_df["Dataset"] = new_labels
else:
    new_feature_df = new_data_df

# Save processed new data
new_feature_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "new_features.csv"), index=False)

# Separate reference and current data
print("Generating data drift report...")
reference_data = old_feature_df.drop(columns=["Dataset"])
current_data = new_feature_df.drop(columns=["Dataset"])

# Generate a data drift report
report = Report(metrics=[DataDriftTable()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html(os.path.join(PROCESSED_DATA_DIR, "data_drift.html"))
print("Data drift report saved.")

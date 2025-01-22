import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from google.cloud import storage
import kagglehub

# Define input and output paths
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET")  # GCS bucket name
IMG_SIZE = (128, 128)  # Resize images to this size

def download_from_kaggle():
    """
    Download the dataset from Kaggle and structure it correctly.
    """
    # Specify the dataset name from Kaggle
    dataset_name = "harishkumardatalab/food-image-classification-dataset"
    
    # Specify the directory paths
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"
    
    # Ensure both directories exist; create them if they don't
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    # Download the dataset
    print("Downloading dataset...")
    path = kagglehub.dataset_download(dataset_name)
    
    print("Dataset downloaded successfully!")
    print("Dataset files are located at:", path)

    # Dynamically find the correct dataset folder
    dataset_root = path
    print("Searching for dataset root...")

    while True:
        subfolders = [os.path.join(dataset_root, d) for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
        if len(subfolders) == 1:  # If there's only one folder, go deeper
            dataset_root = subfolders[0]
            print(f"Found nested folder: {dataset_root}")
        else:  # If we find multiple folders or files, stop searching
            break

    # Move the dataset folder to 'data/raw'
    print("Moving dataset to 'data/raw'...")
    target_path = os.path.join(raw_data_dir, os.path.basename(dataset_root))

    # Move the dataset folder (or file) to the target directory
    shutil.move(dataset_root, target_path)
    print(f"Dataset moved to: {target_path}")
    
    return target_path



def create_directories(base_dir):
    """
    Create necessary directories for processed data.
    """
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)


def upload_to_gcs(local_folder: str, bucket_name: str, gcs_folder: str):
    """
    Uploads processed data to GCS.
    """
    print(f"Uploading data from {local_folder} to gs://{bucket_name}/{gcs_folder}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            gcs_path = os.path.join(gcs_folder, relative_path)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")


def process_and_split_data():
    """
    Process and split the dataset into train/val/test sets.
    """
    raw_data_dir = "data/raw/Food Classification dataset"
    processed_data_dir = "data/processed"
    # Check if the dataset exists in the expected path
    if not os.path.exists(raw_data_dir):
        raise ValueError(f"Dataset not found at {raw_data_dir}. Please download it first.")

    # Gather all image paths and labels
    all_images = []
    labels = []
    
    print("Starting to process data...")
    for class_name in os.listdir(raw_data_dir):
        class_path = os.path.join(raw_data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                all_images.append(img_path)
                labels.append(class_name)

    # Validate dataset size
    if len(all_images) < 3:
        raise ValueError(
            f"Insufficient data for splitting. Found only {len(all_images)} samples. "
            "Ensure the dataset has at least 3 samples for train, val, and test splits."
        )

    # Split into train, val, and test sets
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        all_images, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # Define splits
    splits = {
        "train": (train_images, train_labels),
        "val": (val_images, val_labels),
        "test": (test_images, test_labels)
    }

    # Process and save images
    create_directories(processed_data_dir)
    for split, (images, split_labels) in splits.items():
        print(f"Processing {split} data...")
        for img_path, label in tqdm(zip(images, split_labels), total=len(images)):
            label_dir = os.path.join(processed_data_dir, split, label)
            os.makedirs(label_dir, exist_ok=True)

            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            img.save(os.path.join(label_dir, os.path.basename(img_path)))

    # Upload processed data to GCS

import subprocess

def upload_to_gcs_with_gsutil(local_folder: str, bucket_name: str, gcs_folder: str):
    """
    Upload a local folder to GCS using `gsutil` for faster performance.
    """
    gcs_path = f"gs://{bucket_name}/{gcs_folder}"
    print(f"Uploading {local_folder} to {gcs_path} using gsutil...")
    try:
        subprocess.run(["gsutil", "-m", "cp", "-r", local_folder, gcs_path], check=True)
        print(f"Upload to {gcs_path} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during upload to GCS: {e}")

if __name__ == "__main__":
    # Ensure dataset is downloaded
    if not os.path.exists("data/raw/Food Classification dataset"):
        download_from_kaggle()
    
    # Process the data
    process_and_split_data()
    print("Data processing complete!")

    # Upload processed data to GCS
    upload_to_gcs_with_gsutil("data/processed", GCS_BUCKET_NAME, "data/processed")
    print("Data upload to GCS complete!")

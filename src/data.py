import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from google.cloud import storage

# Define input and output paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET")
GCS_RAW_FOLDER = "data/raw"
GCS_PROCESSED_FOLDER = "data/processed"
IMG_SIZE = (128, 128)  # Resize images to this size


def download_from_gcs(bucket_name: str, gcs_folder: str, local_folder: str):
    """
    Download data from GCS to a local directory.
    """
    print(f"Downloading data from gs://{bucket_name}/{gcs_folder} to {local_folder}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=gcs_folder)
    for blob in blobs:
        relative_path = os.path.relpath(blob.name, gcs_folder)
        local_path = os.path.join(local_folder, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")


def upload_to_gcs(local_folder: str, bucket_name: str, gcs_folder: str):
    """
    Upload processed data to GCS.
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


def create_directories(base_dir):
    """
    Create necessary directories for processed data.
    """
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)


def process_and_split_data():
    """
    Process and split the dataset into train/val/test sets and upload to GCS.
    """
    # Download raw data from GCS
    print("Downloading raw data...")
    download_from_gcs(GCS_BUCKET_NAME, GCS_RAW_FOLDER, RAW_DATA_DIR)

    # Gather all image paths and labels
    all_images = []
    labels = []

    print("Processing raw data...")
    for class_name in os.listdir(RAW_DATA_DIR):
        class_path = os.path.join(RAW_DATA_DIR, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                all_images.append(img_path)
                labels.append(class_name)

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

    # Process and save images locally
    create_directories(PROCESSED_DATA_DIR)
    for split, (images, split_labels) in splits.items():
        print(f"Processing {split} data...")
        for img_path, label in tqdm(zip(images, split_labels), total=len(images)):
            label_dir = os.path.join(PROCESSED_DATA_DIR, split, label)
            os.makedirs(label_dir, exist_ok=True)

            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            img.save(os.path.join(label_dir, os.path.basename(img_path)))

    # Upload processed data to GCS
    print("Uploading processed data to GCS...")
    upload_to_gcs(PROCESSED_DATA_DIR, GCS_BUCKET_NAME, GCS_PROCESSED_FOLDER)
    print("Processed data uploaded to GCS.")


if __name__ == "__main__":
    if not GCS_BUCKET_NAME:
        raise ValueError("Environment variable 'GCS_BUCKET' is not set.")
    
    process_and_split_data()
    print("Data processing and upload complete!")

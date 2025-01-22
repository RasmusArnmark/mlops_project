import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from google.cloud import storage
import kagglehub

# Define input and output paths
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")  # Default to local raw data path
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")  # Default to local processed data path
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET")  # GCS bucket name
IMG_SIZE = (128, 128)  # Resize images to this size
KAGGLE_DATASET_NAME = "harishkumardatalab/food-image-classification-dataset"  # Kaggle dataset identifier

def download_from_kaggle():
    """
    Download the dataset from Kaggle if it's not already downloaded.
    """
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Dataset not found. Downloading from Kaggle...")
        path = kagglehub.dataset_download(KAGGLE_DATASET_NAME)
        dataset_name = os.path.basename(path)
        target_path = os.path.join(RAW_DATA_DIR, dataset_name)
        # Move the dataset folder (or file) to the target directory
        shutil.move(path, target_path)
        print(f"Dataset moved to: {target_path}")
        print(f"Dataset downloaded to {RAW_DATA_DIR}")
    else:
        print(f"Dataset already exists at {RAW_DATA_DIR}")

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
    # Gather all image paths and labels
    all_images = []
    labels = []
    
    print("Starting to process data...")
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

    # Process and save images
    create_directories(PROCESSED_DATA_DIR)
    for split, (images, split_labels) in splits.items():
        print(f"Processing {split} data...")
        for img_path, label in tqdm(zip(images, split_labels), total=len(images)):
            label_dir = os.path.join(PROCESSED_DATA_DIR, split, label)
            os.makedirs(label_dir, exist_ok=True)

            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            img.save(os.path.join(label_dir, os.path.basename(img_path)))

    # Upload to GCS if running in the cloud
    if GCS_BUCKET_NAME:
        print("Uploading processed data to GCS...")
        upload_to_gcs(PROCESSED_DATA_DIR, GCS_BUCKET_NAME, "data/processed")
        print("Processed data uploaded to GCS.")

if __name__ == "__main__":
    # Check if the dataset exists, if not, download it
    download_from_kaggle()
    
    # Process the data
    process_and_split_data()
    print("Data processing complete!")

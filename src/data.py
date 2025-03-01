import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import kagglehub
import subprocess

# Define input and output paths
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
IMG_SIZE = (128, 128)  # Resize images to this size


def download_from_kaggle():
    """
    Download the dataset from Kaggle and structure it correctly.
    """
    dataset_name = "harishkumardatalab/food-image-classification-dataset"
    raw_data_dir = "raw_tmp"
    os.makedirs(raw_data_dir, exist_ok=True)

    print("Downloading dataset...")
    path = kagglehub.dataset_download(dataset_name)
    print("Dataset downloaded successfully!")
    print("Dataset files are located at:", path)

    dataset_root = path
    print("Searching for dataset root...")

    while True:
        subfolders = [
            os.path.join(dataset_root, d)
            for d in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, d))
        ]
        if len(subfolders) == 1:
            dataset_root = subfolders[0]
            print(f"Found nested folder: {dataset_root}")
        else:
            break

    print("Copying dataset to 'raw_tmp'...")
    target_path = os.path.join(raw_data_dir, os.path.basename(dataset_root))
    shutil.copytree(dataset_root, target_path, dirs_exist_ok=True)
    print(f"Dataset copied to: {target_path}")
    return target_path


def create_directories(base_dir):
    """
    Create necessary directories for processed data without deleting existing ones.
    """
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)


def process_and_split_data(
    raw_data_dir="raw_tmp/Food Classification dataset",
    processed_data_dir="processed_tmp",
):
    """
    Process and split the dataset into train/val/test sets.
    """
    if not os.path.exists(raw_data_dir):
        raise ValueError(f"Dataset not found at {raw_data_dir}. Please download it first.")

    all_images, labels = [], []
    print("Starting to process data...")
    for class_name in os.listdir(raw_data_dir):
        class_path = os.path.join(raw_data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                all_images.append(img_path)
                labels.append(class_name)

    if len(all_images) < 3:
        raise ValueError(
            f"Insufficient data for splitting. Found only {len(all_images)} samples. "
            "Ensure the dataset has at least 3 samples for train, val, and test splits."
        )

    train_images, temp_images, train_labels, temp_labels = train_test_split(
        all_images, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    splits = {
        "train": (train_images, train_labels),
        "val": (val_images, val_labels),
        "test": (test_images, test_labels),
    }

    create_directories(processed_data_dir)
    for split, (images, split_labels) in splits.items():
        print(f"Processing {split} data...")
        for img_path, label in tqdm(zip(images, split_labels), total=len(images)):
            label_dir = os.path.join(processed_data_dir, split, label)
            os.makedirs(label_dir, exist_ok=True)

            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            img.save(os.path.join(label_dir, os.path.basename(img_path)))


def move_to_data_dir():
    """
    Move temporary directories into the data directory.
    """
    os.makedirs("data", exist_ok=True)

    if os.path.exists("raw_tmp"):
        shutil.move("raw_tmp", "data/raw")
    if os.path.exists("processed_tmp"):
        shutil.move("processed_tmp", "data/processed")
    print("Moved raw and processed directories into 'data/'")


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
    os.makedirs("raw_tmp", exist_ok=True)
    os.makedirs("processed_tmp", exist_ok=True)

    if not os.listdir("raw_tmp"):
        download_from_kaggle()

    process_and_split_data()
    print("Data processing complete!")

    move_to_data_dir()

    if GCS_BUCKET_NAME:
        upload_to_gcs_with_gsutil("data/processed", GCS_BUCKET_NAME, "data/processed")
        print("Data upload to GCS complete!")
    else:
        print("GCS_BUCKET_NAME is not set. Skipping data upload to GCS.")

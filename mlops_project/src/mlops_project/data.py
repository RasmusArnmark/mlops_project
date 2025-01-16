import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random 

# Define input and output paths
RAW_DATA_DIR = "data/raw"  # Path to the extracted Kaggle dataset
PROCESSED_DATA_DIR = "data/processed"
IMG_SIZE = (128, 128)  # Resize images to this size

def create_directories(base_dir):
    """
    Create necessary directories for processed data.
    """
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)

def process_and_split_data(test_subset_size=500):
    """
    Process and split the dataset into train/val/test sets.
    Use only a subset of the test data if specified.
    """
    # Gather all image paths and labels
    all_images = []
    labels = []

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

    # Use a subset of test data
    if test_subset_size < len(test_images):
        test_subset = random.sample(list(zip(test_images, test_labels)), test_subset_size)
        test_images, test_labels = zip(*test_subset)

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

if __name__ == "__main__":
    process_and_split_data(test_subset_size=500)  # Use a subset of 500 test images
    print("Data processing complete!")

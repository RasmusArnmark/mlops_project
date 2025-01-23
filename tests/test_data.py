import os
from pathlib import Path
from PIL import Image
import shutil
import pytest
from src.data import process_and_split_data

# Constants
TEST_RAW_DIR = "tests/test_image"  # Directory with test images
TEST_PROCESSED_DIR = "tests/test_processed"  # Directory for processed images
IMG_SIZE = (128, 128)  # Expected image size after processing
EXPECTED_SPLITS = ["train", "val", "test"]  # Expected splits
EXPECTED_CLASSES = ["class1", "class2"]  # Replace with actual class names


def setup_test_environment():
    """
    Prepare the test environment by ensuring clean test_processed directory.
    """
    # Ensure the processed directory is empty
    if os.path.exists(TEST_PROCESSED_DIR):
        shutil.rmtree(TEST_PROCESSED_DIR)
    os.makedirs(TEST_PROCESSED_DIR, exist_ok=True)


def validate_split_distribution():
    """
    Ensure all splits (train, val, test) exist and have at least one image.
    """
    for split in EXPECTED_SPLITS:
        split_dir = Path(TEST_PROCESSED_DIR) / split
        assert split_dir.exists(), f"Split directory {split} is missing."
        assert sum(1 for _ in split_dir.rglob("*.jpg")) > 0, f"No images found in {split} split."


def validate_class_directories():
    """
    Ensure all expected classes exist within each split.
    """
    for split in EXPECTED_SPLITS:
        for class_name in EXPECTED_CLASSES:
            class_dir = Path(TEST_PROCESSED_DIR) / split / class_name
            assert class_dir.exists(), f"Class directory {class_name} is missing in {split} split."


def validate_image_processing():
    """
    Ensure all processed images have the correct size.
    """
    for split in EXPECTED_SPLITS:
        for img_path in Path(TEST_PROCESSED_DIR).rglob("*.jpg"):
            img = Image.open(img_path)
            assert img.size == IMG_SIZE, f"Image {img_path} has incorrect size: {img.size}"


def test_process_and_split_data():
    """
    Full test for the process_and_split_data function.
    """
    setup_test_environment()

    # Run the processing function
    process_and_split_data(raw_data_dir=TEST_RAW_DIR, processed_data_dir=TEST_PROCESSED_DIR)

    # Validate results
    validate_split_distribution()
    validate_class_directories()
    validate_image_processing()

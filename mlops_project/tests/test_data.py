import os
from pathlib import Path
from PIL import Image
import pytest
from src.data import process_and_split_data

PROCESSED_DIR = "data/processed"  # Directory where processed data is saved
IMG_SIZE = (128, 128)  # Expected image size after processing


def count_images_in_split(split_name):
    """
    Count the number of images in a specific split (train, val, or test).
    """
    split_path = Path(PROCESSED_DIR) / split_name
    return sum(1 for _ in split_path.rglob("*.jpg"))


def validate_images_in_split(split_name):
    """
    Validate that all images in the specified split have the expected size.
    """
    split_path = Path(PROCESSED_DIR) / split_name
    for img_path in split_path.rglob("*.jpg"):
        img = Image.open(img_path)
        assert img.size == IMG_SIZE, f"Image {img_path} has incorrect size: {img.size}"


def validate_class_directories(split_name, expected_classes):
    """
    Validate that all expected class directories exist in the split.
    """
    split_path = Path(PROCESSED_DIR) / split_name
    for class_name in expected_classes:
        class_dir = split_path / class_name
        assert class_dir.exists(), f"Class directory {class_name} is missing in {split_name}"


def test_processed_data():
    """
    Test the output of process_and_split_data.
    """
    # Ensure processed directory exists
    assert os.path.exists(PROCESSED_DIR), f"{PROCESSED_DIR} does not exist. Run process_and_split_data first."

    # Define expected class names (mock data)
    expected_classes = ["apple_pie", "Baked Potato"]

    # Validate train split
    validate_class_directories("train", expected_classes)
    validate_images_in_split("train")
    assert count_images_in_split("train") > 0, "No images found in train split"

    # Validate val split
    validate_class_directories("val", expected_classes)
    validate_images_in_split("val")
    assert count_images_in_split("val") > 0, "No images found in val split"

    # Validate test split
    validate_class_directories("test", expected_classes)
    validate_images_in_split("test")
    assert count_images_in_split("test") > 0, "No images found in test split"

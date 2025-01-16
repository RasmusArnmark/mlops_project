import os
from pathlib import Path
import tempfile
from mlops_project.src.mlops_project.data import process_and_split_data

def test_process_and_split_data():
    """
    Test the data processing and splitting function using a temporary directory.
    """
    # Create a temporary directory for raw and processed data
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_data_path = os.path.join(temp_dir, "raw")
        processed_data_path = os.path.join(temp_dir, "processed")
        os.makedirs(raw_data_path, exist_ok=True)

        # Create mock raw data structure
        class_dirs = ["class1", "class2"]
        for class_dir in class_dirs:
            os.makedirs(os.path.join(raw_data_path, class_dir), exist_ok=True)
            for i in range(3):  # Add 3 sample images per class
                with open(os.path.join(raw_data_path, class_dir, f"image_{i}.jpg"), "w") as f:
                    f.write("mock_image_content")

        # Run data processing with test paths
        process_and_split_data()

        # Assertions
        processed_dir = Path(processed_data_path)
        assert processed_dir.exists(), "Processed directory was not created"
        assert (processed_dir / "train").exists(), "Train split was not created"
        assert (processed_dir / "val").exists(), "Validation split was not created"
        assert (processed_dir / "test").exists(), "Test split was not created"

        # Check train/val/test splits contain data
        for split in ["train", "val", "test"]:
            split_path = processed_dir / split
            assert len(list(split_path.rglob("*.jpg"))) > 0, f"No images found in {split}"

        # The temporary directory and its contents are automatically cleaned up

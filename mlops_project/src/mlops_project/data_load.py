import pandas as pd
import logging
import kagglehub
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def download_and_prepare_dataset(kaggle_dataset_name, output_file, nrows=None):
    """
    Downloads a dataset from Kaggle and saves it after cleaning.
    
    Args:
        kaggle_dataset_name (str): Kaggle dataset name (e.g., 'username/dataset-name').
        output_file (str): Path to save the cleaned dataset.
        nrows (int, optional): Number of rows to process (for testing).

    Returns:
        str: Path to the cleaned dataset.
    """
    logging.info(f"Downloading dataset: {kaggle_dataset_name}")
    path = kagglehub.dataset_download(kaggle_dataset_name)
    dataset_file_path = os.path.join(path, "en-fr.csv")
    
    if not os.path.exists(dataset_file_path):
        raise FileNotFoundError(f"Expected file 'en-fr.csv' not found in {path}")
    
    logging.info(f"Reading dataset: {dataset_file_path}")
    df = pd.read_csv(dataset_file_path, nrows=nrows)
    logging.info(f"Loaded {len(df)} rows.")

    # Clean the dataset
    def is_relevant(row):
        source = str(row["en"]) if not pd.isnull(row["en"]) else ""
        target = str(row["fr"]) if not pd.isnull(row["fr"]) else ""
        if "««««" in source or "»»»»" in source:
            return False
        if "««««" in target or "»»»»" in target:
            return False
        return True

    df = df[df.apply(is_relevant, axis=1)]
    logging.info(f"Cleaned dataset size: {len(df)} rows.")

    # Save the cleaned dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logging.info(f"Cleaned dataset saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Define paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    raw_data_path = os.path.join(project_root, "data/raw/en-fr.csv")

    # Download and clean the dataset
    download_and_prepare_dataset(
        kaggle_dataset_name="dhruvildave/en-fr-translation-dataset",
        output_file=raw_data_path,
        nrows=250000,  # Limit for testing
    )

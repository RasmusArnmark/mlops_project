import pandas as pd
from datasets import Dataset
from transformers import MBart50Tokenizer
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def preprocess_data(
    file_path,
    percentage=10,
    tokenizer_name="facebook/mbart-large-50",
    max_len=128,
    chunk_size=1000,
    batch_size=32,
    save_path=None,
):
    """
    Tokenizes the dataset and saves the processed output.

    Args:
        file_path (str): Path to the cleaned CSV file.
        percentage (float): Percentage of data to subsample (default: 10%).
        tokenizer_name (str): Tokenizer model to use.
        max_len (int): Max token length for tokenization.
        chunk_size (int): Number of rows to load at once.
        batch_size (int): Batch size for tokenization.
        save_path (str): Path to save the processed dataset.

    Returns:
        Dataset: A tokenized dataset.
    """
    tokenizer = MBart50Tokenizer.from_pretrained(tokenizer_name)

    def load_data(file_path, chunk_size=10000):
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk = chunk.rename(columns={"en": "source", "fr": "target"})
            chunks.append(Dataset.from_pandas(chunk))
            logging.info(f"Loaded chunk of size {len(chunk)}.")
        if len(chunks) > 1:
            dataset = Dataset.from_dict(pd.concat([chunk.to_pandas() for chunk in chunks]).to_dict(orient="list"))
        elif len(chunks) == 1:
            dataset = chunks[0]
        else:
            logging.error("No valid data found.")
            return None
        return dataset

    def subsample_data(dataset, percentage):
        if not (0 < percentage <= 100):
            raise ValueError("Percentage must be between 0 and 100.")
        num_samples = int(len(dataset) * (percentage / 100))
        subsampled_dataset = dataset.shuffle(seed=42).select(range(num_samples))
        logging.info(f"Subsampled dataset to {num_samples} examples.")
        return subsampled_dataset

    def tokenize_batch(batch):
        source_texts = [str(x) for x in batch["source"]]
        target_texts = [str(x) for x in batch["target"]]
        tokenized_source = tokenizer(source_texts, max_length=max_len, truncation=True, padding="max_length")
        tokenized_target = tokenizer(target_texts, max_length=max_len, truncation=True, padding="max_length")
        return {
            "input_ids": tokenized_source["input_ids"],
            "attention_mask": tokenized_source["attention_mask"],
            "labels": tokenized_target["input_ids"],
        }

    dataset = load_data(file_path, chunk_size)
    if dataset is None:
        return None
    logging.info(f"Dataset loaded with {len(dataset)} examples.")

    subsampled_dataset = subsample_data(dataset, percentage)
    tokenized_dataset = subsampled_dataset.map(tokenize_batch, batched=True, batch_size=batch_size)

    if save_path:
        tokenized_dataset.save_to_disk(save_path)
        logging.info(f"Tokenized dataset saved to: {save_path}")

    return tokenized_dataset


if __name__ == "__main__":
    # Define paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    raw_data_path = os.path.join(project_root, "data/raw/en-fr.csv")
    processed_data_path = os.path.join(project_root, "data/processed/tokenized_dataset")

    # Preprocess the dataset
    tokenized_data = preprocess_data(
        raw_data_path,
        percentage=100,
        max_len=128,
        chunk_size=10000,
        batch_size=32,
        save_path=processed_data_path,
    )
    if tokenized_data:
        print(tokenized_data[0])

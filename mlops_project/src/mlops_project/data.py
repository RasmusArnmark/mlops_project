import pandas as pd
from datasets import Dataset
from transformers import MBartTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def load_data(file_path, source_col="source", target_col="target", chunk_size=10000):
    """
    Loads a CSV file into a Hugging Face Dataset in manageable chunks.

    Args:
        file_path (str): Path to the CSV file.
        source_col (str): Column name for the source language (default: 'source').
        target_col (str): Column name for the target language (default: 'target').
        chunk_size (int): Number of rows to load per chunk.

    Returns:
        Dataset: A Hugging Face Dataset object with 'source' and 'target' columns.
    """
    # Load CSV in chunks
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        if source_col not in chunk.columns or target_col not in chunk.columns:
            raise ValueError(f"The CSV file must contain '{source_col}' and '{target_col}' columns.")
        chunk = chunk.rename(columns={source_col: "source", target_col: "target"})
        chunks.append(Dataset.from_pandas(chunk))
        logging.info(f"Loaded chunk of size {len(chunk)}.")

    # Concatenate all chunks into a single Dataset
    if len(chunks) > 1:
        dataset = Dataset.from_dict(pd.concat([chunk.to_pandas() for chunk in chunks]).to_dict(orient="list"))
    elif len(chunks) == 1:
        dataset = chunks[0]
    else:
        raise ValueError("No data found in the CSV file.")
    
    return dataset

def subsample_data(dataset, percentage=10):
    """
    Subsamples a percentage of the dataset.

    Args:
        dataset (Dataset): The input dataset.
        percentage (float): The percentage of data to keep (0-100).

    Returns:
        Dataset: A subsampled dataset.
    """
    if not (0 < percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100.")

    # Calculate the number of samples to keep
    num_samples = int(len(dataset) * (percentage / 100))

    # Shuffle and select the subset
    subsampled_dataset = dataset.shuffle(seed=42).select(range(num_samples))
    logging.info(f"Subsampled dataset to {num_samples} examples.")

    return subsampled_dataset

def tokenize_data(dataset, tokenizer_name="facebook/mbart-large-50", max_len=128, batch_size=32):
    """
    Tokenizes the input dataset for use with mBART.

    Args:
        dataset (Dataset): A dataset object containing 'source' and 'target' fields.
        tokenizer_name (str): The pre-trained tokenizer to use (default: mBART).
        max_len (int): Maximum sequence length for tokenized data.
        batch_size (int): Batch size for tokenization.

    Returns:
        Dataset: A tokenized dataset.
    """
    # Load the tokenizer
    tokenizer = MBartTokenizer.from_pretrained(tokenizer_name)

    # Tokenization function
    def tokenize_batch(batch):
        # Tokenize source and target texts
        tokenized_source = tokenizer(
            batch["source"],
            max_length=max_len,
            truncation=True,
            padding="max_length",
        )
        tokenized_target = tokenizer(
            batch["target"],
            max_length=max_len,
            truncation=True,
            padding="max_length",
        )

        # Combine tokenized inputs
        return {
            "input_ids": tokenized_source["input_ids"],
            "attention_mask": tokenized_source["attention_mask"],
            "labels": tokenized_target["input_ids"],
        }

    # Apply tokenization in batches
    tokenized_dataset = dataset.map(tokenize_batch, batched=True, batch_size=batch_size)
    logging.info("Tokenization complete.")

    return tokenized_dataset

def preprocess_data(file_path, percentage=10, tokenizer_name="facebook/mbart-large-50", max_len=128, chunk_size=10000, batch_size=32):
    """
    Loads, subsamples, and tokenizes the dataset.

    Args:
        file_path (str): Path to the CSV file.
        percentage (float): Percentage of data to subsample (default: 10%).
        tokenizer_name (str): The pre-trained tokenizer to use.
        max_len (int): Maximum sequence length for tokenized data.
        chunk_size (int): Number of rows to load at a time.
        batch_size (int): Batch size for tokenization.

    Returns:
        Dataset: A tokenized and optionally subsampled dataset.
    """
    # Load the dataset in chunks
    dataset = load_data(file_path, source_col="en", target_col="fr", chunk_size=chunk_size)

    # Subsample the dataset
    subsampled_dataset = subsample_data(dataset, percentage)

    # Tokenize the dataset
    tokenized_dataset = tokenize_data(subsampled_dataset, tokenizer_name, max_len, batch_size)

    return tokenized_dataset

if __name__ == "__main__":
    import os

    # Define file paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_path = os.path.join(project_root, "data/raw/en-fr.csv")

    # Preprocess the dataset
    tokenized_data = preprocess_data(data_path, percentage=20, max_len=128, chunk_size=10000, batch_size=32)

    # Example output: Inspect the first tokenized item
    print(tokenized_data[0])

from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import typer

app = typer.Typer()

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path
        self.data = pd.read_csv(raw_data_path)  # Load dataset as a DataFrame

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.data.iloc[index]

    def subsample(self, sample_size: int) -> None:
        """Create a subsample of the dataset."""
        self.data = self.data.sample(n=sample_size, random_state=42)

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        if 'text' in self.data.columns:
            self.data['text'] = self.data['text'].str.lower()  # Example processing
        output_path = output_folder / "processed_data.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")


@app.command("preprocess")
def preprocess(
    raw_data_path: Path = typer.Option(..., help="Path to the raw data file"),
    output_folder: Path = typer.Option(..., help="Directory to save the processed data"),
    sample_size: int = typer.Option(1000, help="Number of rows to include in the subsample")
) -> None:
    """
    Preprocess the data and create a subsample.

    Args:
        raw_data_path (Path): Path to the raw data file.
        output_folder (Path): Directory where the processed data will be saved.
        sample_size (int): Number of rows to include in the subsample.
    """
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    total_rows = len(dataset)
    print(f"Total rows in dataset: {total_rows}")
    sample_size = min(sample_size, total_rows)
    print(f"Creating a subsample of size: {sample_size}")
    dataset.subsample(sample_size)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    app()

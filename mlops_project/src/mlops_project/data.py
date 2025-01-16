import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import torch
import typer
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


# Define your transforms (for preprocessing)
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a fixed size
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize (ImageNet stats)
    ])


def preprocess_data(raw_dir: str= 'data/raw', processed_dir: str= 'data/processed') -> None:
    """Process raw data and save it to processed directory."""
    # Prepare data directory paths
    food_dir = Path(raw_dir)
    
    # Define a transform for preprocessing the images
    transform = get_transforms()

    dataset = datasets.ImageFolder(root=food_dir, transform=transform)

    # Split the dataset into train/test sets (80/20 split)
    train_data, test_data = train_test_split(dataset.samples, test_size=0.2, random_state=42)

    # Create DataLoader for train and test sets
    train_dataset = torch.utils.data.Subset(dataset, range(len(train_data)))
    test_dataset = torch.utils.data.Subset(dataset, range(len(test_data)))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Optionally save data
    torch.save(train_dataset, f"{processed_dir}/train_data.pt")
    torch.save(test_dataset, f"{processed_dir}/test_data.pt")


def food_images() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for food images."""
    train_loader = torch.load("data/processed/train_data.pt")
    test_loader = torch.load("data/processed/test_data.pt")
    return train_loader, test_loader



if __name__ == "__main__":
    #typer.run(preprocess_data)
    preprocess_data()
    train, _ = food_images()
    print(train)


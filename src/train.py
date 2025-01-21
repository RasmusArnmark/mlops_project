import os
import subprocess
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import FoodCNN
import wandb

# Hyperparameters
IMG_SIZE = (128, 128)

def ensure_data_available(data_dir: str):
    """
    Ensure data is available locally by running `dvc pull`.
    """
    if not os.path.exists(data_dir):
        print(f"Running `dvc pull` to fetch data into {data_dir}...")
        subprocess.run(["dvc", "pull"], check=True)
        print("Data fetched successfully with DVC.")
    else:
        print(f"Data already exists at {data_dir}.")

def train_model(data_dir: str, model_dir: str, batch_size: int, learning_rate: float, epochs: int):
    # Ensure data is available
    ensure_data_available(data_dir)

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    run = wandb.init(
        project="food-classification",
        config={"BATCH_SIZE": batch_size, "LEARNING_RATE": learning_rate, "EPOCHS": epochs},
    )

    # Define transforms for the data
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load datasets
    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    num_classes = len(train_data.classes)
    model = FoodCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = correct_train / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = correct_val / total_val

        # Log metrics for each epoch
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        })

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")

    # Save the model locally
    model_path = os.path.join(model_dir, "food_cnn.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved locally to {model_path}")

    # Log model as a W&B artifact
    artifact = wandb.Artifact(
        name="food_classifier",
        type="model",
        description="A model trained to classify food images",
        metadata={"Val Accuracy": val_accuracy},
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--model-dir", type=str, default="models/")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    train_model(args.data_dir, args.model_dir, args.batch_size, args.learning_rate, args.epochs)

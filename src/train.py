import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from google.cloud import storage
from model import FoodCNN
import wandb

# Hyperparameters
IMG_SIZE = (128, 128)
GCS_BUCKET_NAME = "foodclassrae"  # Replace with your GCS bucket name
GCS_PROCESSED_DATA_FOLDER = "data/processed"
GCS_MODEL_FOLDER = "models"


def download_from_gcs(bucket_name: str, gcs_folder: str, local_folder: str):
    """
    Download data from GCS to a local directory.
    """
    print(f"Downloading data from gs://{bucket_name}/{gcs_folder} to {local_folder}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=gcs_folder)
    for blob in blobs:
        relative_path = os.path.relpath(blob.name, gcs_folder)
        local_path = os.path.join(local_folder, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")


def upload_to_gcs(local_path: str, bucket_name: str, gcs_path: str):
    """
    Upload a file to GCS.
    """
    print(f"Uploading {local_path} to gs://{bucket_name}/{gcs_path}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")


def train_model(data_dir: str, model_dir: str, batch_size: int, learning_rate: float, epochs: int):
    """
    Train the FoodCNN model using the specified dataset, hyperparameters, and save it to the specified directory.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        download_from_gcs(GCS_BUCKET_NAME, GCS_PROCESSED_DATA_FOLDER, data_dir)

    os.makedirs(model_dir, exist_ok=True)

    run = wandb.init(
        project="food-classification",
        config={"BATCH_SIZE": batch_size, "LEARNING_RATE": learning_rate, "EPOCHS": epochs},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_data.classes)
    model = FoodCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

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

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = correct_val / total_val

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        })

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {running_loss / len(train_loader):.4f}, "
            f"Val Loss: {val_loss / len(val_loader):.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

    model_path = os.path.join(model_dir, "food_cnn.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved locally to {model_path}")

    gcs_model_path = os.path.join(GCS_MODEL_FOLDER, "food_cnn.pth")
    upload_to_gcs(model_path, GCS_BUCKET_NAME, gcs_model_path)

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

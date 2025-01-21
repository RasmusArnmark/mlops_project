import os
import subprocess
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from google.cloud import storage
from model import FoodCNN
import wandb
import typer

# Hyperparameters
IMG_SIZE = (128, 128)
LOCAL_DATA_DIR = "data/processed"
GCS_BUCKET_NAME = "foodclassrae"  # Your bucket
GCS_DATA_DIR = f"gs://{GCS_BUCKET_NAME}/data/processed"
MODEL_DIR = "models"

# Helper Functions
def download_gcs_data(gcs_path: str, local_path: str):
    """
    Download data from GCS to the local directory.
    """
    print(f"Downloading data from {gcs_path} to {local_path}...")
    subprocess.run(["gsutil", "-m", "cp", "-r", gcs_path, local_path], check=True)
    print("Data downloaded successfully.")

def upload_to_gcs(local_path: str, gcs_path: str):
    """
    Upload a file to GCS.
    """
    print(f"Uploading {local_path} to {gcs_path}...")
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to {gcs_path}.")

# Main Training Function
def train_model(BATCH_SIZE: int = 64, LEARNING_RATE: float = 0.001, EPOCHS: int = 3):
    # Determine data path based on environment
    if os.getenv("GCS_BUCKET"):  # Running in GCP
        if not os.path.exists(LOCAL_DATA_DIR):
            os.makedirs(LOCAL_DATA_DIR)
        download_gcs_data(GCS_DATA_DIR, LOCAL_DATA_DIR)
        DATA_DIR = LOCAL_DATA_DIR
    else:  # Local environment
        DATA_DIR = LOCAL_DATA_DIR

    # Ensure the model directory exists
    print("Model is training")
    os.makedirs(MODEL_DIR, exist_ok=True)

    run = wandb.init(
        project="food-classification",
        config={"BATCH_SIZE": BATCH_SIZE, "LEARNING_RATE": LEARNING_RATE, "EPOCHS": EPOCHS},
    )

    # Define transforms for the data
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Load datasets
    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss function, and optimizer
    num_classes = len(train_data.classes)
    model = FoodCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
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

        print(f"Epoch {epoch + 1}/{EPOCHS}, "
              f"Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")

    # Determine the model path
    model_path = os.path.join(MODEL_DIR, "food_cnn.pth")

    # Save the trained model locally
    torch.save(model.state_dict(), model_path)
    print(f"Model saved locally to {model_path}")

    # Upload the model to GCS
    if os.getenv("GCS_BUCKET"):  # Running in GCP
        gcs_model_path = f"models/{os.path.basename(model_path)}"
        upload_to_gcs(model_path, gcs_model_path)

    # Save model as a W&B artifact
    artifact = wandb.Artifact(
        name="food_classifier",
        type="model",
        description="A model trained to classify food images",
        metadata={"Val Accuracy": val_accuracy},
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)


if __name__ == "__main__":
    typer.run(train_model)

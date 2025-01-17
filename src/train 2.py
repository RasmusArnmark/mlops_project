import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import FoodCNN
import wandb
import typer

# Hyperparameters
<<<<<<< HEAD:mlops_project/src/train.py
=======
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
>>>>>>> fa7891b2a366561bf397ab1d6cf0db93ad6225f2:src/train 2.py
IMG_SIZE = (128, 128)
DATA_DIR = "data/processed"
MODEL_DIR = "models"

def get_new_model_path(base_path, base_name="food_cnn", extension=".pth"):
    """Generate a new model file path if one already exists."""
    counter = 1
    new_path = os.path.join(base_path, f"{base_name}{extension}")
    while os.path.exists(new_path):
        new_path = os.path.join(base_path, f"{base_name}_{counter}{extension}")
        counter += 1
    return new_path

def train_model(BATCH_SIZE: int = 264, LEARNING_RATE: float = 0.001, EPOCHS: int = 10):
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
    model_path = get_new_model_path(MODEL_DIR, base_name="food_cnn", extension=".pth")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

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

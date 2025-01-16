import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import FoodCNN

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
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


def train_model():
    # Ensure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Define transforms for the data
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Load datasets
    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)

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
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {correct/total:.4f}")

    # Determine the model path
    model_path = get_new_model_path(MODEL_DIR, base_name="food_cnn", extension=".pth")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model()

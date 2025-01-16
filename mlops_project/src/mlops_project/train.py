import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import FoodCNN
import wandb

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
IMG_SIZE = (128, 128)
DATA_DIR = "data/processed"
MODEL_PATH = "models/food_cnn.pth"

wandb.login()

def train_model():
    # Initialize wandb
    wandb.init(
        project="food-classification", 
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "model": "FoodCNN",
        },
    )
    config = wandb.config

    transform = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])
    
    # Load datasets
    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    num_classes = len(train_data.classes)
    model = FoodCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Track gradients and model
    wandb.watch(model, criterion, log="all", log_freq=10)
    print("Model training started...")
    # Training loop
    for epoch in range(config.epochs):
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

        # Log metrics to wandb
        wandb.log({
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "val_accuracy": correct / total,
            "epoch": epoch + 1,
        })

        print(f"Epoch {epoch+1}/{config.epochs}, "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {correct/total:.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    train_model()

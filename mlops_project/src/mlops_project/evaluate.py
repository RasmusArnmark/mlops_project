import torch
import typer
from model import FoodCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str = 'models/food_cnn.pth') -> None:
    """Evaluate a trained model."""
    print("Evaluating the model")
    print(f"Loading model checkpoint: {model_checkpoint}")
    
    # W&B initialization
    wandb.init(
        project="food-classification", 
        config={
            "batch_size": 32,  # Default batch size
            "img_size": (128, 128),  # Default image size
        },
    )
    config = wandb.config
    
    # Define the dataset transformation
    transform = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])
    
    DATA_DIR = "data/processed"
    test_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)
    
    # Load the model
    model = FoodCNN(num_classes=len(test_data.classes)).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    
    # Evaluation
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for img, target in test_loader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).sum().item()
            total += target.size(0)
    
    # Calculate and log accuracy
    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f}")
    wandb.log({"test_accuracy": accuracy})
    
    # Finalize W&B logging
    wandb.finish()


if __name__ == "__main__":
    typer.run(evaluate)

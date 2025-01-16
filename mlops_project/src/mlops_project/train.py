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
MODEL_PATH = "models/food_cnn.pth"

def train_model():
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

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

<<<<<<< HEAD
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train and save the model
trainer.train()
model.save_pretrained(model_output_path)
tokenizer.save_pretrained(model_output_path)

artifact = wandb.Artifact(
    name="corrupt_mnist_model",
    type="model",
    description="A model trained to classify corrupt MNIST images",
    metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
)
artifact.add_file("model.pth")
run.log_artifact(artifact)


# Finish the WandB run
wandb.finish()
=======
if __name__ == "__main__":
    train_model()
>>>>>>> ef182d32a7cad9636cd1c2b1d491bc957404fae9

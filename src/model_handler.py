import torch
import torch.nn.functional as F

def load_model(model_path: str):
    """
    Load the PyTorch model from a file.
    """
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()  # Set the model to evaluation mode
    return model

def predict_image(image_tensor, model):
    """
    Predict the class of the image using the loaded model.
    
    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor.
        model: Loaded PyTorch model.
        
    Returns:
        str: Predicted class label.
    """
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0))  # Add batch dimension
        probabilities = F.softmax(outputs, dim=1)
        class_idx = torch.argmax(probabilities, dim=1).item()
    return f"class_{class_idx}"  # Replace with actual class mapping if available

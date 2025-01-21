import torch
from torchvision import transforms
from PIL import Image

def preprocess_image(image: Image.Image):
    """
    Preprocess the input image for the PyTorch model.
    
    Args:
        image (PIL.Image): Input image.
        
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # Define preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),         # Convert to Tensor
        transforms.Normalize(          # Normalize based on dataset stats
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform(image)

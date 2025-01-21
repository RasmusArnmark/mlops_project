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
        transforms.Resize((128, 128)),  # Match training input size
        transforms.ToTensor(),         # Convert to Tensor
        transforms.Normalize(          # Normalize to match training stats
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5]
        ),
    ])
    return transform(image)

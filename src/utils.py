import torch
from torchvision import transforms
from PIL import Image

def preprocess_image(image: Image.Image):
    """
    Preprocess the input image for the PyTorch model.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        torch.Tensor: Preprocessed image tensor with shape [1, 3, 128, 128].
    """
    # Define preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to match model input size
        transforms.ToTensor(),         # Convert image to Tensor
    ])

    # Apply transformations
    tensor = transform(image)  # Shape: [3, 128, 128]

    # Add batch dimension here
    tensor = tensor.unsqueeze(0)  # Shape: [1, 3, 128, 128]
    return tensor


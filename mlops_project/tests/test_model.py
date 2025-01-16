import torch
from mlops_project.src.mlops_project.model import FoodCNN

def test_food_cnn_forward_pass():
    """
    Test a forward pass through the FoodCNN model.
    """
    num_classes = 10
    model = FoodCNN(num_classes=num_classes)

    # Create mock input: batch of 8 images with 3 channels, 128x128
    input_tensor = torch.randn(8, 3, 128, 128)

    # Perform forward pass
    output = model(input_tensor)

    # Assertions
    assert output.shape == (8, num_classes), "Output shape mismatch"
    assert torch.is_tensor(output), "Output is not a tensor"
    assert not torch.isnan(output).any(), "Output contains NaN values"

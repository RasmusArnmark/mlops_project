import os
import torch
import torch.nn.functional as F
from google.cloud import storage
from src.model import FoodCNN

def download_model_from_gcs(bucket_name: str, gcs_model_path: str, local_model_path: str):
    try:
        print(f"Downloading model from gs://{bucket_name}/{gcs_model_path} to {local_model_path}...")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_model_path)

        os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
        blob.download_to_filename(local_model_path)
        print(f"Model downloaded to {local_model_path}.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise


def load_model(model_path: str = "models/food_cnn.pth", bucket_name: str = "foodclassrae"):
    """
    Load the PyTorch model, either from a local path or GCS if running in the cloud.

    Args:
        model_path (str): Path to the model file.
        bucket_name (str): Name of the GCS bucket (optional).

    Returns:
        PyTorch model: Loaded model in evaluation mode.
    """
    # Check if model exists locally
    if not os.path.exists(model_path):
        if bucket_name:
            print("Model not found locally. Attempting to download from GCS...")
            gcs_model_path = os.path.basename(model_path)
            download_model_from_gcs(bucket_name, model_path, model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    # Recreate the model architecture
    print(f"Recreating model architecture for {model_path}...")
    model = FoodCNN(num_classes=len(CLASS_MAPPING))  # Ensure num_classes matches CLASS_MAPPING

    # Load the state dictionary
    print(f"Loading state dictionary from {model_path}...")
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")
    return model


def predict_image(image_tensor, model, class_mapping=None):
    """
    Predict the class of an image using the loaded model.

    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor.
        model: Loaded PyTorch model.
        class_mapping (dict): Optional mapping of class indices to labels.

    Returns:
        str: Predicted class label or class index.
    """
    with torch.no_grad():
        # Pass the tensor directly to the model (batch dimension already added)
        outputs = model(image_tensor)  # No need for unsqueeze here
        probabilities = F.softmax(outputs, dim=1)
        class_idx = torch.argmax(probabilities, dim=1).item()

    # Use class mapping if available, else return class index
    if class_mapping:
        return class_mapping.get(class_idx, f"Unknown class {class_idx}")
    return f"class_{class_idx}"


# Example class mapping (update as per your model's output)
CLASS_MAPPING = {
    0: 'Baked Potato',
    1: 'Crispy Chicken',
    2: 'Donut',
    3: 'Fries',
    4: 'Hot Dog',
    5: 'Sandwich',
    6: 'Taco',
    7: 'Taquito',
    8: 'apple_pie',
    9: 'burger',
    10: 'butter_naan',
    11: 'chai',
    12: 'chapati',
    13: 'cheesecake',
    14: 'chicken_curry',
    15: 'chole_bhature',
    16: 'dal_makhani',
    17: 'dhokla',
    18: 'fried_rice',
    19: 'ice_cream',
    20: 'idli',
    21: 'jalebi',
    22: 'kaathi_rolls',
    23: 'kadai_paneer',
    24: 'kulfi',
    25: 'masala_dosa',
    26: 'momos',
    27: 'omelette',
    28: 'paani_puri',
    29: 'pakode',
    30: 'pav_bhaji',
    31: 'pizza',
    32: 'samosa',
    33: 'sushi',
}

if __name__ == "__main__":
    # Example Usage
    model = load_model("models/food_cnn_3.pth", bucket_name=os.getenv("GCS_BUCKET"))
    print(predict_image(torch.rand(3, 128, 128), model, class_mapping=CLASS_MAPPING))

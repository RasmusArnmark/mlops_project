import gradio as gr
from PIL import Image
import torch
from src.model_handler import load_model, predict_image, CLASS_MAPPING
from src.utils import preprocess_image

# Load the model
model_path = "models/food_cnn.pth"  # Adjust the path as needed
model = load_model(model_path)

# Define the function Gradio will use
def gradio_interface(image):
    """
    Function for Gradio to handle an image input and return a prediction.
    """
    try:
        # Directly preprocess the PIL image
        processed_image = preprocess_image(image)

        # Predict using the loaded model
        prediction = predict_image(processed_image, model, CLASS_MAPPING)

        return f"Predicted Class: {prediction}"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Set up Gradio interface
gr.Interface(
    fn=gradio_interface,  # Function to handle input/output
    inputs=gr.Image(type="pil"),  # Accepts PIL Image objects
    outputs="text",  # Outputs text (predicted class)
    title="Food Image Classification",
    description="Upload a food image, and the model will classify it into one of the food categories."
).launch()


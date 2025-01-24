import gradio as gr
from src.model_handler import load_model, predict_image, CLASS_MAPPING
from src.utils import preprocess_image
from prometheus_client import Counter, Histogram, start_http_server
import time

# Prometheus metrics
REQUEST_COUNT = Counter("request_count", "Number of prediction requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Latency of prediction requests")
ERROR_COUNT = Counter("error_count", "Number of failed prediction requests")

# Start Prometheus metrics server
start_http_server(8001)  # Port for metrics (accessible at /metrics)

# Load the model
model_path = "models/food_cnn.pth"  # Adjust the path as needed
model = load_model(model_path, bucket_name="foodclassrae")


def gradio_interface(image):
    """
    Function for Gradio to handle an image input and return a prediction.
    """
    REQUEST_COUNT.inc()  # Increment request count
    start_time = time.time()  # Start timer to measure latency

    try:
        # Preprocess the PIL image
        processed_image = preprocess_image(image)

        # Predict using the loaded model
        prediction = predict_image(processed_image, model, CLASS_MAPPING)

        latency = time.time() - start_time  # Calculate request latency
        REQUEST_LATENCY.observe(latency)  # Log latency to Prometheus

        return f"Predicted Class: {prediction}"
    except Exception as e:
        ERROR_COUNT.inc()  # Increment error count
        return f"Error during prediction: {str(e)}"


# Set up Gradio interface
gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Food Image Classification",
    description="Upload a food image, and the model will classify it into one of the food categories.",
).launch(server_port=8080, server_name="0.0.0.0")

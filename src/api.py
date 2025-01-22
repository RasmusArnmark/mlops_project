from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import os
from src.model_handler import load_model, predict_image, CLASS_MAPPING
from src.utils import preprocess_image

app = FastAPI()

# Load the model during startup
model_path = os.getenv("MODEL_PATH", "models/food_cnn.pth")
model = load_model(model_path)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Food Image Classification API!"}

@app.get("/health/")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess the image
        image = Image.open(BytesIO(await file.read()))
        processed_image = preprocess_image(image)
    except Exception as e:
        return JSONResponse(content={"error": f"Image preprocessing failed: {str(e)}"}, status_code=400)

    try:
        # Run prediction
        prediction = predict_image(processed_image, model, class_mapping=CLASS_MAPPING)

        # Save the data and prediction
        
    except Exception as e:
        return JSONResponse(content={"error": f"Model prediction failed: {str(e)}"}, status_code=500)

    return {"filename": file.filename, "prediction": prediction}

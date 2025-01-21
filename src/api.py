from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from model_handler import load_model, predict_image
from utils import preprocess_image

app = FastAPI()

# Load the model during startup
model = load_model("../models/food_cnn_3.pth")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Food Image Classification API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess the image
        image = Image.open(BytesIO(await file.read()))
        processed_image = preprocess_image(image)

        # Run prediction
        prediction = predict_image(processed_image, model)

        return {"filename": file.filename, "prediction": prediction}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

import logging
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import os
from fastapi import UploadFile, File, HTTPException, Request
from utils import preprocess_image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import httpx
import re
import os
import subprocess

from dotenv import load_dotenv

load_dotenv(".env")

# Configure environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize FastAPI app
app = FastAPI(title="Scanalyze Analysis APP")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# https://www.kaggle.com/models/aliamrali/chest-xray-classification/keras/v1

@app.on_event("startup")
async def startup_event():
    """Initialize all models at startup"""
    try:
        logging.info("Loading all models at startup...")

        # Load kidney model
        app.state.kidney_model = load_model(
            "C:/Users/khale/.cache/kagglehub/models/aliamrali/kidney-classification/keras/v1/1/resnet50_kidney_ct_augmented.h5")
        # app.state.kidney_model = load_model_from_kaggle(
        #     "aliamrali",
        #     "kidney-classification",
        #     "v1",
        #     "resnet50_kidney_ct_augmented.h5"
        # )
        logging.info("Kidney model loaded successfully")

        # Load Chest model
        app.state.chest_model = load_model(
            "C:/Users/khale/.cache/kagglehub/models/aliamrali/chest-xray-classification/keras/v1/1/resnet50_lung_xray.h5")
        # app.state.chest_model = load_model_from_kaggle(
        #     "aliamrali",
        #     "chest-xray-classification",
        #     "v1",
        #     "resnet50_lung_xray.h5"
        # )
        logging.info("Chest model loaded successfully")

    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise


async def get_gemini_report(disease_class: str) -> str:
    """
    Calls Gemini API to get a concise medical report about the predicted disease class.
    """

    if disease_class == CLASS_LABELS[1]:
        return "No Diseases Detected :)"

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    prompt = (
        f"Provide a concise, clear, and medically accurate report for a patient diagnosed with: {disease_class}. "
        "Describe what this disease is, its typical symptoms, and recommended next steps for the patient. "
        "Keep the report short and easy to understand."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        response = await client.post(GEMINI_URL, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"Error: {response.status_code}, {response.text}"


@app.post("/")
def health():
    return {"message": "Scanalyze API"}


# Class labels for predictions
CLASS_LABELS = {0: "could be Covid",
                1: "No abnormal findings detected",
                2: "could be Pneumonia",
                3: "could be Tuberculosis"}


@app.post("/chest-predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Endpoint to predict tuberculosis from uploaded chest X-ray
    Args:
        file: Uploaded image file
        request: FastAPI request object to access app state
    Returns:
        dict: Prediction results including class and confidence
    """
    try:

        # Get model from app state
        chest_model = request.app.state.chest_model
        if chest_model is None:
            raise ValueError("Chest model not initialized")

        # Validate image file
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        contents = await file.read()

        # Read and preprocess image
        img = Image.open(BytesIO(contents)).convert("RGB")

        img_array = preprocess_image(img)

        # Make prediction
        # prediction = chest_model.predict(img_array, verbose=0)
        # predicted_class = int(prediction[0][0] > 0.5)
        # confidence = float(prediction[0][0])

        prediction = chest_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        result = CLASS_LABELS[predicted_class]

        # After predicting the disease class:
        report = await get_gemini_report(result)

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result,
            "report": report,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    return resnet_preprocess_input(img_array)


# Endpoint for Kidney Classification
kidney_labels = ["Cyst", "Normal", "Stone", "Tumor"]


@app.post("/kidney-predict")
async def predict_kidney(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()

        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = preprocess_image(img)

        kidney_model = request.app.state.kidney_model

        if kidney_model is None:
            raise ValueError("Kidney model is not loaded")

        prediction = kidney_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        result = kidney_labels[predicted_class]

        # After predicting the disease class:
        report = await get_gemini_report(result)

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result,
            "report": report
        }
    except Exception as e:
        print(f"Kidney predict error: {e}")
        return {"success": False, "error": str(e)}


# === RUN FASTAPI + STREAMLIT TOGETHER ===


def run_streamlit():
    subprocess.run(["streamlit", "run", "scan_st.py"])


if __name__ == "__main__":

    import uvicorn
    import threading

    # Run Streamlit in a separate thread
    threading.Thread(target=run_streamlit, daemon=True).start()

    # Run FastAPI backend
    uvicorn.run(app, host="localhost", port=8088)

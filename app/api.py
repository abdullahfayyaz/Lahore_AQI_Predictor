# app/api.py
from fastapi import FastAPI
import sys
import os

# Allow importing from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.predict import make_prediction

app = FastAPI(title="Lahore AQI API", version="1.0")

@app.get("/")
def home():
    return {"message": "AQI Predictor API is Running 🚀"}

@app.get("/predict")
def predict():
    """
    Triggers the ML pipeline to predict live AQI.
    """
    result = make_prediction()
    return result

# Run with: uvicorn app.api:app --reload
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import create_model

# Load the exported model and its info once the API starts
with open("model/model_info.json") as f:
    MODEL_INFO = json.load(f)
FEATURES = MODEL_INFO["features"]
model = joblib.load("model/model.joblib")

# Input validation: a request must contain all 87 features, each one a number
URLFeatures = create_model("URLFeatures", **{name: (float, ...) for name in FEATURES})

app = FastAPI(title="Phishing Detection API")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_INFO["model_name"],
        "version": MODEL_INFO["version"],
    }


@app.post("/predict")
def predict(features: URLFeatures):
    # Turn the request into a one-row table, with the columns in the training order
    row = pd.DataFrame([features.model_dump()])[FEATURES]
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return {
        "prediction": "phishing" if prediction == 1 else "legitimate",
        "phishing_probability": round(probability, 3),
    }
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("../models/xgb_v1.pkl")

@app.post("/predict")
def predict(data: dict):
    # Convert input JSON to DataFrame
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}

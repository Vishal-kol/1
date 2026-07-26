import joblib
import pandas as pd
import json

def init():
    global model
    model = joblib.load("models/xgb_v1.pkl")

def run(raw_data):
    data = json.loads(raw_data)
    df = pd.DataFrame(data["instances"])
    preds = model.predict(df)
    return {"predictions": preds.tolist()}

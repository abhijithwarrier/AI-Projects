from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load("iris_model.joblib")

app = FastAPI(title="Iris Prediction API")

# Pydantic model for JSON input
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Iris ML Model API is running!"}

@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.features).reshape(1, -1)
    pred = model.predict(arr).tolist()
    return {"prediction": pred}

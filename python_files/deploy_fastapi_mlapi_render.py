from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("../models/iris_model.joblib")

app = FastAPI(title="Deployed ML API")

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.features).reshape(1, -1)
    pred = model.predict(arr).tolist()
    return {"prediction": pred}

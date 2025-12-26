import logging
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

model = joblib.load("../models/iris_model.joblib")

app = FastAPI(title="ML API with Logging")

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)

    logger.info(f"Received input features: {data.features}")

    prediction = model.predict(features).tolist()

    logger.info(f"Model prediction: {prediction}")

    return {"prediction": prediction}

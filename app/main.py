import os
import pandas as pd
import mlflow
import mlflow.sklearn

from fastapi import FastAPI
from app.schemas import MachineInput

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load the registered production candidate
model = mlflow.sklearn.load_model(
    "models:/predictive-maintenance-model/1"
)

app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine failure prediction API",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "predictive-maintenance-model",
        "version": 1,
    }


@app.post("/predict")
def predict(machine: MachineInput):

    data = pd.DataFrame(
        [
            {
                "Type": machine.type,
                "Air temperature [K]": machine.air_temperature,
                "Process temperature [K]": machine.process_temperature,
                "Rotational speed [rpm]": machine.rotational_speed,
                "Torque [Nm]": machine.torque,
                "Tool wear [min]": machine.tool_wear,
            }
        ]
    )

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    return {
        "failure_probability": round(probability, 4),
        "prediction": prediction,
        "status": "FAILURE_RISK" if prediction == 1 else "HEALTHY",
    }
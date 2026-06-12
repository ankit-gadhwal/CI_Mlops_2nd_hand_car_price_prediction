import mlflow
import dagshub
import pandas as pd
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

# ----------------------------------------
# MLflow / DagsHub Setup
# ----------------------------------------

dagshub_url = "https://dagshub.com"
repo_owner = "ankit-gadhwal"
repo_name = "CI_Mlops_2nd_hand_car_price_prediction"

mlflow.set_tracking_uri(
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)

# ----------------------------------------
# Load Production Model
# ----------------------------------------

def load_model():
    client = mlflow.tracking.MlflowClient()

    versions = client.get_latest_versions(
        "model",
        stages=["Production"]
    )

    if not versions:
        raise Exception(
            "No model found in Production stage."
        )

    run_id = versions[0].run_id

    print(f"Loading model from run: {run_id}")

    return mlflow.pyfunc.load_model(
        f"runs:/{run_id}/model"
    )

model = load_model()

# ----------------------------------------
# FastAPI App
# ----------------------------------------

app = FastAPI(
    title="2nd Hand Car Price Prediction API"
)

# ----------------------------------------
# Request Schema
# ----------------------------------------

class CarInput(BaseModel):
    on_road_old: float
    on_road_now: float
    years: float
    km: float
    rating: float
    condition: float
    economy: float
    top_speed: float
    hp: float
    torque: float

# ----------------------------------------
# Routes
# ----------------------------------------

@app.get("/")
def home():
    return {
        "message": "Car Price Prediction API Running"
    }

@app.post("/predict")
def predict(data: CarInput):

    # depreciation = (
    #     data.on_road_old - data.on_road_now
    # )

    km_per_year = (
        data.km / (data.years + 1)
    )

    performance = (
        data.hp + data.torque
    )

    value_score = (
        data.economy * data.rating
    )

    input_data = pd.DataFrame([{
        "on road old": data.on_road_old,
        "on road now": data.on_road_now,
        "years": data.years,
        "km": data.km,
        "rating": data.rating,
        "condition": data.condition,
        "economy": data.economy,
        "top speed": data.top_speed,
        "hp": data.hp,
        "torque": data.torque,
        # "depreciation": depreciation,
        "km_per_year": km_per_year,
        "performance": performance,
        "value_score": value_score
    }])

    prediction = model.predict(
        input_data
    )

    pred_value = float(
        np.array(prediction).flatten()[0]
    )

    return {
        "predicted_price": pred_value
    }
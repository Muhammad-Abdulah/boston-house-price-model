from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# ----- Safe Path -----
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# ----- Load Model -----
data = joblib.load(MODEL_PATH)
model = data["model"]
FEATURES = data["features"]

# ----- App -----
app = FastAPI(title="House Price Prediction API")

# ----- CORS FIX (IMPORTANT FOR FRONTEND) -----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all frontend requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- Input Schema ---------------
class HouseInput(BaseModel):
    rm: float
    lstat: float
    ptratio: float
    nox: float
    dis: float
    tax: float
    age: float

# --------------- Root Route ---------------
@app.get("/")
def home():
    return {"message": "API is running"}

# --------------- Prediction Route ---------------
@app.post("/predict")
def predict(data: HouseInput):

    input_df = pd.DataFrame([[
        data.rm,
        data.lstat,
        data.ptratio,
        data.nox,
        data.dis,
        data.tax,
        data.age
    ]], columns=FEATURES)

    prediction = model.predict(input_df)[0]

    return {
        "predicted_price": float(prediction)
    }
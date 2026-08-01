# Import libraries
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Initialize FastAPI application
app = FastAPI(title="NovaPay Fraud Detection API")

# Load the saved pipeline
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR
    / "notebook"
    / "model"
    / "novapay_xgboost_pipeline.pkl"
)

model = joblib.load(MODEL_PATH)

# Define transaction input schema
class Transaction(BaseModel):
    home_country: str = Field(examples=["US"])
    source_currency: str = Field(examples=["USD"])
    dest_currency: str = Field(examples=["EUR"])
    channel: str = Field(examples=["Mobile"])

    amount_src: float = Field(examples=[250.0])
    amount_usd: float = Field(examples=[270.0])
    fee: float = Field(examples=[3.5])
    exchange_rate_src_to_dest: float = Field(examples=[0.92])

    new_device: bool = Field(examples=[True])
    ip_country: str = Field(examples=["US"])
    location_mismatch: bool = Field(examples=[False])

    ip_risk_score: float = Field(examples=[25.0])
    kyc_tier: str = Field(examples=["Standard"])
    account_age_days: int = Field(examples=[365])
    device_trust_score: float = Field(examples=[80.0])
    chargeback_history_count: int = Field(examples=[0])
    risk_score_internal: float = Field(examples=[20.0])
    txn_velocity_1h: int = Field(examples=[1])
    txn_velocity_24h: int = Field(examples=[4])
    corridor_risk: float = Field(examples=[10.0])

    year: float = Field(examples=[2026])
    month: float = Field(examples=[7])
    day_of_month: float = Field(examples=[31])
    day_of_week: str = Field(examples=["Thursday"])
    hour: float = Field(examples=[13])

    is_weekend: int = Field(examples=[0])
    ip_address_missing: int = Field(examples=[0])

# Define home endpoint
@app.get("/")
def home():
    return {
        "message": "NovaPay Fraud Detection API is running"
    }

# Define prediction endpoint
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    transaction_df = pd.DataFrame(
        [transaction.model_dump()]
    )

    prediction = int(model.predict(transaction_df)[0])

    fraud_probability = float(
        model.predict_proba(transaction_df)[0, 1]
    )

    return {
        "prediction": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": round(fraud_probability, 4)
    }
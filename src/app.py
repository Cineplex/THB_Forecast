from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import sys

# Add src to python path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deploy_models import DeployModel

app = FastAPI(
    title="THB Forecast API (XGBoost)",
    description="API for forecasting THB/USD exchange rates using XGBoost",
    version="1.1.0"
)

# Global variable for the model
model_inference = None

@app.on_event("startup")
def load_model():
    global model_inference
    try:
        # Construct path to the model file
        # Assuming app.py is in src/ and model is in src/ds/models/save_models/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "ds", "models", "save_models", "xgboost_model.pkl")
        
        if os.path.exists(model_path):
            model_inference = DeployModel(model_path)
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️ Model file not found at {model_path}. Prediction endpoint will fail.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

class PredictionInput(BaseModel):
    gold: float = Field(..., description="Gold Price")
    oil: float = Field(..., description="Oil Price")
    bond_yield: float = Field(..., description="US 10Y Bond Yield")
    dxy: float = Field(..., description="US Dollar Index")
    sp500: float = Field(..., description="S&P 500 Index")
    set_index: float = Field(..., description="SET Index")
    rsi: float = Field(..., description="RSI Indicator")
    macd: float = Field(..., description="MACD Indicator")
    pct_change: float = Field(..., description="Percentage Change")
    volatility_5: float = Field(..., description="Volatility (5 days)")
    volatility_20: float = Field(..., description="Volatility (20 days)")
    gold_oil_ratio: float = Field(..., description="Gold/Oil Ratio")
    bond_dxy_ratio: float = Field(..., description="Bond/DXY Ratio")
    dist_sma20: float = Field(..., description="Distance from SMA20")
    lag_1: float = Field(..., description="Previous Day Price (THB/USD)")
    lag_7: float = Field(..., description="Price 7 Days Ago")
    day_of_week: int = Field(..., description="Day of Week (0-6)")
    month: int = Field(..., description="Month (1-12)")
    is_holiday_th: int = Field(..., description="Is Thai Holiday (0 or 1)")

    class Config:
        schema_extra = {
            "example": {
                "gold": 2000.0,
                "oil": 80.0,
                "bond_yield": 4.5,
                "dxy": 105.0,
                "sp500": 4500.0,
                "set_index": 1400.0,
                "rsi": 50.0,
                "macd": 0.1,
                "pct_change": 0.001,
                "volatility_5": 0.2,
                "volatility_20": 0.3,
                "gold_oil_ratio": 25.0,
                "bond_dxy_ratio": 0.04,
                "dist_sma20": 0.5,
                "lag_1": 35.5,
                "lag_7": 35.2,
                "day_of_week": 2,
                "month": 11,
                "is_holiday_th": 0
            }
        }

@app.get("/")
def read_root():
    return {
        "message": "Welcome to THB Forecast API (XGBoost).",
        "status": "online",
        "model_loaded": model_inference is not None
    }

@app.get("/health")
def health_check():
    if model_inference is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    try:
        # Convert Pydantic model to dict
        data_dict = input_data.dict()
        
        # Make prediction
        result = model_inference.predict_next_day(data_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
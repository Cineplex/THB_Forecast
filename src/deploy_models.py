import joblib
import pandas as pd
import os
import numpy as np

class DeployModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.features = None
        self.metadata = None
        self.load_model()

    def load_model(self):
        """Loads the XGBoost model from the .pkl file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        try:
            print(f"Loading model from {self.model_path}...")
            model_package = joblib.load(self.model_path)
            self.model = model_package['model']
            self.features = model_package['features']
            self.metadata = model_package.get('metadata', {})
            print("✅ Model loaded successfully!")
            print(f"   Type: {self.metadata.get('model_type', 'Unknown')}")
            print(f"   Test MAPE: {self.metadata.get('test_mape', 0) * 100:.4f}%")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def predict_next_day(self, input_data: dict) -> dict:
        """
        Predicts the next day's price based on input features.
        
        Args:
            input_data (dict): Dictionary containing all required features.
            
        Returns:
            dict: Prediction result including predicted price, diff, and metadata.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")

        # Convert input dict to DataFrame
        df = pd.DataFrame([input_data])
        
        # Determine expected features from the model itself if possible
        expected_features = self.features
        if hasattr(self.model, 'feature_names_in_'):
            expected_features = self.model.feature_names_in_
        
        # Ensure all required features are present (checking against expected_features)
        # We only check for features that are actually expected by the model
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
            
        # Select and order features matches training
        # Only keep features that the model expects
        X = df[expected_features].copy()
        
        # Handle Categorical Features (must match training)
        cat_cols = ['day_of_week', 'month', 'is_holiday_th']
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype('category')

        # Predict Diff
        try:
            pred_diff = self.model.predict(X)[0]
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
        
        # Calculate Price = Lag1 + Diff
        lag_1 = input_data.get('lag_1')
        if lag_1 is None:
             raise ValueError("Feature 'lag_1' is required for price reconstruction.")
             
        pred_price = lag_1 + pred_diff
        
        return {
            "predicted_price": float(pred_price),
            "predicted_diff": float(pred_diff),
            "lag_1": float(lag_1),
            "model_metadata": self.metadata
        }

# Singleton instance for easy import
# You can initialize this in app.py startup event
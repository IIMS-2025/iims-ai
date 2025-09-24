from fastapi import HTTPException
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict

class ModelService:
    """
    Model loading and prediction service
    """
    
    def __init__(self, model_path: str = "artifacts/models/model_CatBoost.joblib"):
        self.model = None
        self.model_path = Path(model_path)
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load the trained CatBoost model"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
                
                # If model has feature_names_ attribute, use it
                if hasattr(self.model, 'feature_names_'):
                    self.feature_columns = self.model.feature_names_
                else:
                    # Define expected features (should match training)
                    self.feature_columns = [
                        'tenant_id', 'location_id', 'product_id', 'current_price', 'year', 'month', 'day',
                        'week_of_year', 'quarter', 'dow_encoded', 'is_weekend', 'is_holiday',
                        'sales_last_7_day', 'sales_lag_1', 'sales_lag_3', 'sales_lag_7', 'sales_lag_14',
                        'sales_rolling_mean_3', 'sales_rolling_std_3', 'sales_rolling_mean_7', 'sales_rolling_std_7',
                        'sales_rolling_mean_14', 'sales_rolling_std_14', 'product_category_encoded',
                        'price_category_encoded', 'tenant_location_encoded', 'category_location_encoded',
                        'price_times_holiday', 'price_vs_category_avg'
                    ]
            else:
                print(f"Model file not found at {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def predict(self, features: Dict) -> float:
        """
        Make prediction using loaded model
        
        Returns:
            float: Predicted sales
        """
        
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            # Create feature vector in correct order
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0))
            
            # Convert to DataFrame for compatibility
            X = pd.DataFrame([feature_vector], columns=self.feature_columns)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            return max(0, int(round(prediction)))
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

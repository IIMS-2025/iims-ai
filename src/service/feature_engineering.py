from typing import List, Dict
from datetime import datetime
import numpy as np

class FeatureEngineer:
    """
    Feature engineering service to prepare data for model prediction
    """
    
    @staticmethod
    def prepare_prediction_features(
        location_id: int,
        tenant_id: int, 
        product_id: int,
        product_category: str,
        current_price: float,
        prediction_date: datetime,
        historical_sales: List[int]
    ) -> Dict:
        """
        Create feature set for model prediction
        This should match the features used during model training
        """
        
        # ===== TEMPORAL FEATURES =====
        features = {
            'tenant_id': tenant_id,
            'location_id': location_id,
            'product_id': product_id,
            'current_price': current_price,
            'year': prediction_date.year,
            'month': prediction_date.month,
            'day': prediction_date.day,
            'week_of_year': prediction_date.isocalendar().week,
            'quarter': (prediction_date.month - 1) // 3 + 1,
            'day_of_week': prediction_date.strftime('%A'),
            'dow_encoded': prediction_date.weekday(),
            'is_weekend': int(prediction_date.weekday() >= 5),
            'is_holiday': int(prediction_date.isoweekday() in (6, 7))
        }
        
        # ===== HISTORICAL SALES FEATURES =====
        if len(historical_sales) >= 7:
            features['sales_last_7_day'] = sum(historical_sales[-7:])
            features['sales_rolling_mean_7'] = np.mean(historical_sales[-7:])
            features['sales_rolling_std_7'] = np.std(historical_sales[-7:])
        else:
            features['sales_last_7_day'] = sum(historical_sales)
            features['sales_rolling_mean_7'] = np.mean(historical_sales) if historical_sales else 0
            features['sales_rolling_std_7'] = np.std(historical_sales) if len(historical_sales) > 1 else 0
        
        # Lag features (if enough history)
        for i, lag in enumerate([1, 3, 7, 14]):
            if len(historical_sales) >= lag:
                features[f'sales_lag_{lag}'] = historical_sales[-lag]
            else:
                features[f'sales_lag_{lag}'] = 0
        
        # Rolling statistics for different windows
        for window in [3, 14]:
            if len(historical_sales) >= window:
                features[f'sales_rolling_mean_{window}'] = np.mean(historical_sales[-window:])
                features[f'sales_rolling_std_{window}'] = np.std(historical_sales[-window:])
            else:
                features[f'sales_rolling_mean_{window}'] = features['sales_rolling_mean_7']
                features[f'sales_rolling_std_{window}'] = features['sales_rolling_std_7']
        
        # ===== CATEGORICAL ENCODING =====
        # These should match the label encoders used during training
        category_mappings = {
            'Beverage': 0, 'Appetizer': 1, 'Main Course': 2, 'Dessert': 3
        }
        features['product_category_encoded'] = category_mappings.get(product_category, 0)
        
        # Price features
        features['price_category'] = 'Medium'  # Simplified
        features['price_category_encoded'] = 2
        
        # Interaction features  
        features['tenant_location'] = f"{tenant_id}_{location_id}"
        features['category_location'] = f"{product_category}_{location_id}"
        features['tenant_location_encoded'] = hash(f"{tenant_id}_{location_id}") % 1000
        features['category_location_encoded'] = hash(f"{product_category}_{location_id}") % 1000
        
        features['price_times_holiday'] = current_price * features['is_holiday']
        features['price_vs_category_avg'] = 1.0  # Simplified
        
        return features

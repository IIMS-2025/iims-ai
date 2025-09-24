from pydantic import BaseModel, Field
from typing import List

class ProductInfo(BaseModel):
    """Product information for prediction request"""
    product_id: int = Field(..., description="Product ID")
    product_category: str = Field(..., description="Product category")
    current_price: float = Field(..., description="Current price of the product")
    historical_sales: List[int] = Field(..., description="Historical sales data for the product (5 days)")

class PredictionRequest(BaseModel):
    """Request model for sales prediction"""
    tenant_id: int = Field(..., description="Tenant ID for the prediction")
    location_id: int = Field(..., description="Location ID for the prediction")
    date: str = Field(..., description="Date for prediction (YYYY-MM-DD)")
    products: List[ProductInfo] = Field(None, description="List of product informations")

class ProductPrediction(BaseModel):
    """Individual product sales prediction"""
    product_id: int
    product_category: str
    current_price: float
    predicted_sales: int
    
class LocationPrediction(BaseModel):
    """Complete location prediction response"""
    location_id: int
    tenant_id: int
    prediction_date: str
    day_of_week: str
    is_holiday: bool
    total_predicted_sales: int
    products: List[ProductPrediction]
    
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str

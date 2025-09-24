from fastapi import APIRouter, HTTPException
from schemas.prediction import PredictionRequest, LocationPrediction, HealthResponse, ProductPrediction
from service.model import ModelService
from service.feature_engineering import FeatureEngineer
from datetime import datetime
from typing import Dict

router = APIRouter()

# Initialize services
model_service = ModelService()
feature_engineer = FeatureEngineer()

@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sales Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_service.model is not None else "unhealthy",
        model_loaded=model_service.model is not None,
        timestamp=datetime.utcnow().isoformat()
    )

@router.post("/predict", response_model=LocationPrediction)
async def predict_sales(request: PredictionRequest):
    """
    Predict sales for a specific location and date
    
    - **tenant_id**: The tenant ID for multi-tenant support
    - **location_id**: The location ID to predict for
    - **date**: The date for prediction (YYYY-MM-DD format)
    - **products**: List of products with their details
    
    Returns predictions for all products at that location
    """
    try:
        # Parse and validate date
        try:
            prediction_date = datetime.strptime(request.date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Make predictions for each product
        product_predictions = []
        total_predicted_sales = 0

        for product in request.products:
            product_id = product.product_id
            product_category = product.product_category
            current_price = product.current_price
            historical_sales = product.historical_sales

            # Create features
            features = feature_engineer.prepare_prediction_features(
                location_id=request.location_id,
                tenant_id=request.tenant_id,
                product_id=product_id,
                product_category=product_category,
                current_price=current_price,
                prediction_date=prediction_date,
                historical_sales=historical_sales
            )

            # Make prediction
            predicted_sales = model_service.predict(features)
            total_predicted_sales += predicted_sales
            
            # Create product prediction
            product_predictions.append(ProductPrediction(
                product_id=product_id,
                product_category=product_category,
                current_price=current_price,
                predicted_sales=predicted_sales
            ))

        # Create response
        return LocationPrediction(
            location_id=request.location_id,
            tenant_id=request.tenant_id,
            prediction_date=request.date,
            day_of_week=prediction_date.strftime('%A'),
            is_holiday=prediction_date.isoweekday() in (6, 7),  # Placeholder, implement holiday logic as needed
            total_predicted_sales=total_predicted_sales,
            products=product_predictions
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

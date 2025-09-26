# IIMS AI - Sales Prediction Service

An AI-powered sales forecasting service built with FastAPI and machine learning models. This service provides accurate sales predictions for different locations and products using advanced ML algorithms including CatBoost and XGBoost.

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Support](#support)

## Features

- Sales Forecasting: Predict sales for specific locations and dates
- Multi-tenant Support: Support for multiple tenants with isolated data
- ML Models: Uses CatBoost and XGBoost for accurate predictions
- Feature Engineering: Advanced feature engineering for better predictions
- RESTful API: Clean and documented REST API
- Health Monitoring: Built-in health checks and monitoring
- Docker Support: Fully containerized with Docker and Docker Compose
- Data Persistence: Persistent storage for models and data

## Tech Stack

- Backend: FastAPI (Python 3.12)
- ML Models: CatBoost, XGBoost
- Data Processing: Pandas, NumPy
- Containerization: Docker, Docker Compose
- Package Management: UV (ultra-fast Python package manager)
- API Documentation: Swagger/OpenAPI

## Prerequisites

Before running the application, ensure you have the following installed:

- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)
- Git (for cloning the repository)

### System Requirements
- OS: Linux, macOS, or Windows with WSL2
- RAM: Minimum 4GB (8GB recommended)
- Storage: At least 2GB free space

## Quick Start

1. Clone the repository
   ```bash
   git clone https://github.com/IIMS-2025/iims-ai.git
   cd iims-ai
   ```

2. Start the application
   ```bash
   ./start.sh
   ```

3. Access the application
   - API: http://localhost:8080
   - Documentation: http://localhost:8080/docs
   - Health Check: http://localhost:8080/health

## API Documentation

Once the application is running, you can access:

- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc
- OpenAPI JSON: http://localhost:8080/openapi.json

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available endpoints |
| `/health` | GET | Health check endpoint |
| `/predict` | POST | Generate sales predictions |

### Example API Usage

```bash
# Health check
curl http://localhost:8080/health

# Make a prediction
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant1",
    "location_id": "loc001",
    "date": "2025-10-01",
    "products": [
      {
        "product_id": "prod1",
        "category": "electronics",
        "price": 299.99
      }
    ]
  }'
```

## Support

For support and questions:

- Create an issue on GitHub
- Contact the development team

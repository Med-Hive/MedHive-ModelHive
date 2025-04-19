from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class WelcomeMessage(BaseModel):
    message: str = Field(..., description="Welcome message")
    project: str = Field(..., description="Project name")
    version: str = Field(..., description="API version")
    documentation: str = Field(..., description="Documentation URL")
    endpoints: Dict[str, str] = Field(..., description="Available API endpoints")
    repository: str = Field(..., description="Project repository URL")
    notice: str = Field(..., description="Important usage notice")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Welcome to Pneumonia Detection API",
                "project": "MedHive Pneumonia Prediction API",
                "version": "1.0.0",
                "documentation": "/docs",
                "endpoints": {
                    "health_check": "/health",
                    "prediction": "/predict"
                },
                "repository": "https://github.com/your-org/breast-cancer-detection",
                "notice": "This API is intended for research purposes only"
            }
        }

class PredictionInput(BaseModel):
    """This class is empty because we'll use UploadFile for image input in the routes"""
    pass
    
    class Config:
        json_schema_extra = {
            "example": {}
        }

class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="Prediction (0 for normal, 1 for pneumonia)")
    diagnosis: str = Field(..., description="Diagnosis interpretation (Normal or Pneumonia)")
    probability: float = Field(..., description="Probability of pneumonia")
    timestamp: float = Field(..., description="Timestamp of the prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "diagnosis": "Pneumonia",
                "probability": 0.95,
                "timestamp": 1650374400.0
            }
        }

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the service")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    timestamp: float = Field(..., description="Current timestamp")
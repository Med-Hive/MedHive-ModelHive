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
                "message": "Welcome to Breast Cancer Detection API",
                "project": "MedHive Breast Cancer Prediction API",
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
    radius_mean: float = Field(..., description="Mean of distances from center to points on the perimeter")
    texture_mean: float = Field(..., description="Standard deviation of gray-scale values")
    perimeter_mean: float = Field(..., description="Mean size of the core tumor")
    area_mean: float = Field(..., description="Mean area of the core tumor")
    smoothness_mean: float = Field(..., description="Mean of local variation in radius lengths")
    compactness_mean: float = Field(..., description="Mean of perimeter^2 / area - 1.0")
    concavity_mean: float = Field(..., description="Mean of severity of concave portions of the contour")
    concave_points_mean: float = Field(..., description="Mean for number of concave portions of the contour")
    symmetry_mean: float = Field(..., description="Mean symmetry of the cell nucleus")
    fractal_dimension_mean: float = Field(..., description="Mean for 'coastline approximation' - 1")
    radius_se: float = Field(..., description="Standard error for the mean of distances from center to points on the perimeter")
    texture_se: float = Field(..., description="Standard error for standard deviation of gray-scale values")
    perimeter_se: float = Field(..., description="Standard error for mean size of the core tumor")
    area_se: float = Field(..., description="Standard error for mean area of the core tumor")
    smoothness_se: float = Field(..., description="Standard error for mean of local variation in radius lengths")
    compactness_se: float = Field(..., description="Standard error for mean of perimeter^2 / area - 1.0")
    concavity_se: float = Field(..., description="Standard error for mean of severity of concave portions of the contour")
    concave_points_se: float = Field(..., description="Standard error for number of concave portions of the contour")
    symmetry_se: float = Field(..., description="Standard error for mean symmetry of the cell nucleus")
    fractal_dimension_se: float = Field(..., description="Standard error for 'coastline approximation' - 1")
    radius_worst: float = Field(..., description="Worst or largest mean value for distance from center to points on the perimeter")
    texture_worst: float = Field(..., description="Worst or largest mean value for standard deviation of gray-scale values")
    perimeter_worst: float = Field(..., description="Worst or largest mean value for mean size of the core tumor")
    area_worst: float = Field(..., description="Worst or largest mean value for mean area of the core tumor")
    smoothness_worst: float = Field(..., description="Worst or largest mean value for mean of local variation in radius lengths")
    compactness_worst: float = Field(..., description="Worst or largest mean value for mean of perimeter^2 / area - 1.0")
    concavity_worst: float = Field(..., description="Worst or largest mean value for mean of severity of concave portions of the contour")
    concave_points_worst: float = Field(..., description="Worst or largest mean value for number of concave portions of the contour")
    symmetry_worst: float = Field(..., description="Worst or largest mean value for mean symmetry of cell nucleus")
    fractal_dimension_worst: float = Field(..., description="Worst or largest mean value for 'coastline approximation' - 1")
    
    class Config:
        schema_extra = {
            "example": {
                "radius_mean": 17.99,
                "texture_mean": 10.38,
                "perimeter_mean": 122.8,
                "area_mean": 1001.0,
                "smoothness_mean": 0.1184,
                "compactness_mean": 0.2776,
                "concavity_mean": 0.3001,
                "concave_points_mean": 0.1471,
                "symmetry_mean": 0.2419,
                "fractal_dimension_mean": 0.07871,
                "radius_se": 1.095,
                "texture_se": 0.9053,
                "perimeter_se": 8.589,
                "area_se": 153.4,
                "smoothness_se": 0.006399,
                "compactness_se": 0.04904,
                "concavity_se": 0.05373,
                "concave_points_se": 0.01587,
                "symmetry_se": 0.03003,
                "fractal_dimension_se": 0.006193,
                "radius_worst": 25.38,
                "texture_worst": 17.33,
                "perimeter_worst": 184.6,
                "area_worst": 2019.0,
                "smoothness_worst": 0.1622,
                "compactness_worst": 0.6656,
                "concavity_worst": 0.7119,
                "concave_points_worst": 0.2654,
                "symmetry_worst": 0.4601,
                "fractal_dimension_worst": 0.1189
            }
        }

class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="Prediction (0 for benign, 1 for malignant)")
    diagnosis: str = Field(..., description="Diagnosis interpretation (Benign or Malignant)")
    probability: float = Field(..., description="Probability of malignancy")
    timestamp: float = Field(..., description="Timestamp of the prediction")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the service")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    timestamp: float = Field(..., description="Current timestamp")
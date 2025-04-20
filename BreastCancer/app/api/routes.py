from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import time
from app.core.models import PredictionInput, PredictionOutput, WelcomeMessage, HealthResponse
from app.services.prediction import ModelService
from app.core.config import settings
from app.core.exceptions import ModelLoadError, PredictionError, FeatureError

router = APIRouter()

@router.get("/", response_model=WelcomeMessage)
async def root():
    """Root endpoint that provides API information"""
    return {
        "message": "Welcome to Breast Cancer Detection API",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "documentation": "/docs",
        "endpoints": {
            "health_check": "/health",
            "prediction": "/predict"
        }
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint that verifies the API and model status"""
    try:
        model = ModelService.get_model()
        return {
            "status": "ok",
            "model_loaded": model is not None,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "model_loaded": False,
            "timestamp": time.time(),
            "error": str(e)
        }

@router.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Predict breast cancer diagnosis based on input features"""
    try:
        start_time = time.time()
        prediction, probability = ModelService.predict(input_data.model_dump())
        return {
            "prediction": prediction,
            "diagnosis": "Malignant" if prediction else "Benign",
            "probability": probability,
            "timestamp": time.time()
        }
    except (ModelLoadError, PredictionError, FeatureError) as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.error_code,
                "message": e.detail
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": f"An unexpected error occurred: {str(e)}"
            }
        )
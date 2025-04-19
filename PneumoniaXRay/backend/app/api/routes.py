from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import time
import os
import tempfile
from app.core.models import PredictionOutput, WelcomeMessage, HealthResponse
from app.services.prediction import ModelService
from app.core.config import settings
from app.core.exceptions import ModelLoadError, PredictionError, FeatureError

router = APIRouter()

@router.get("/", response_model=WelcomeMessage)
async def root():
    """Root endpoint that provides API information"""
    return {
        "message": "Welcome to Pneumonia Detection API",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "documentation": "/docs",
        "endpoints": {
            "health_check": "/health",
            "prediction": "/predict"
        },
        "repository": "https://github.com/your-org/pneumonia-detection",
        "notice": "This API is intended for research purposes only"
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
async def predict(file: UploadFile = File(...)):
    """
    Predict pneumonia from chest X-ray image
    
    - **file**: Chest X-ray image file (JPEG, PNG)
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_FILE_TYPE",
                    "message": "Only image files are accepted"
                }
            )
        
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Make prediction using the temporary file
            prediction, probability = await ModelService.predict_from_image(temp_file_path)
            
            return {
                "prediction": prediction,
                "diagnosis": "Pneumonia" if prediction else "Normal",
                "probability": probability,
                "timestamp": time.time()
            }
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
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
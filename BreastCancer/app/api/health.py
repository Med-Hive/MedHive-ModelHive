
from fastapi import APIRouter
from app.core.models import HealthResponse
from app.services.prediction import ModelService
import time

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "model_loaded": ModelService.is_ready(),
        "timestamp": time.time()
    }
from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()

@router.get("")
async def root():
    """
    Root endpoint that returns basic API information
    """
    return {
        "name": settings.PROJECT_NAME,
        "version": "1.0.0",
        "description": "LLM-powered symptom analysis system with vector similarity search"
    }
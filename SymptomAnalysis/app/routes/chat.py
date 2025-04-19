from fastapi import APIRouter, Depends, HTTPException
from app.models.requests import SymptomAnalysisRequest
from app.models.responses import SymptomAnalysisResponse, ErrorResponse
from app.services.chat_service import ChatService
from typing import Optional
from pydantic import BaseModel

router = APIRouter()
chat_service = ChatService()

class FeedbackRequest(BaseModel):
    prediction_id: str
    correct_diagnosis: Optional[str] = None
    feedback_text: Optional[str] = None
    severity_reported: Optional[str] = None

@router.post("/analyze", 
             response_model=SymptomAnalysisResponse,
             responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def analyze_symptoms(
    request: SymptomAnalysisRequest,
) -> SymptomAnalysisResponse:
    """
    Analyze symptoms using LLM and vector similarity search
    """
    try:
        return await chat_service.analyze_symptoms(
            symptoms=request.symptoms,
            additional_context=request.additional_context
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing symptoms: {str(e)}"
        )

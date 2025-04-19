from fastapi import APIRouter, Depends, HTTPException
from app.models.requests import SymptomAnalysisRequest
from app.models.responses import SymptomAnalysisResponse, ErrorResponse
from app.services.chat_service import ChatService
from app.core.security import get_current_user
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
    current_user: dict = Depends(get_current_user)
) -> SymptomAnalysisResponse:
    """
    Analyze symptoms using LLM and vector similarity search
    """
    try:
        return await chat_service.analyze_symptoms(
            symptoms=request.symptoms,
            additional_context=request.additional_context,
            user_id=current_user["username"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing symptoms: {str(e)}"
        )

@router.post("/feedback",
            responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def submit_feedback(
    feedback: FeedbackRequest,
    current_user: dict = Depends(get_current_user)
) -> dict:
    """
    Submit feedback for a previous diagnosis
    """
    try:
        await chat_service.record_feedback(
            prediction_id=feedback.prediction_id,
            user_id=current_user["username"],
            correct_diagnosis=feedback.correct_diagnosis,
            feedback_text=feedback.feedback_text,
            severity_reported=feedback.severity_reported
        )
        
        return {
            "status": "success",
            "message": "Feedback recorded successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error recording feedback: {str(e)}"
        )
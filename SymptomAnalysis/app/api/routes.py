"""
API routes for the Symptom Analysis Chatbot
"""
import logging
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.database import get_astra_db, search_diseases_by_symptoms
from app.services.llm import generate_response
from app.services.disease import get_all_symptoms, extract_symptoms
from app.config import DATA_CSV_PATH

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Extract the last user message
        user_message = ""
        for message in reversed(request.history):
            if message.user:
                user_message = message.user
                break
        
        if not user_message:
            return ChatResponse(agent="I don't see a message to respond to. How can I help you with your health concerns?")
        
        # Get all possible symptoms
        all_symptoms = get_all_symptoms(DATA_CSV_PATH)
        
        # Get database connection
        vector_store = get_astra_db()
        
        # Extract symptoms from user message
        detected_symptoms = extract_symptoms(user_message, all_symptoms)
        
        # Search for diseases
        disease_matches = search_diseases_by_symptoms(detected_symptoms, vector_store)
        
        # Create conversation history for the LLM
        conversation_history = ""
        for msg in request.history:
            if msg.user:
                conversation_history += f"User: {msg.user}\n"
            if msg.agent:
                conversation_history += f"Assistant: {msg.agent}\n"
        
        # Generate response
        response = generate_response(disease_matches, conversation_history, user_message)
        
        return ChatResponse(agent=response)
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return ChatResponse(agent="I'm sorry, I encountered an error while processing your request. Please try again later.")

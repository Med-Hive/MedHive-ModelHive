"""
LLM service for generating responses using Groq
"""
import logging
from fastapi import HTTPException
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def get_llm():
    """Initialize and return the LLM"""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set")
        raise HTTPException(status_code=500, detail="LLM API key not configured")
    
    return ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=LLM_TEMPERATURE,
    )

def generate_response(disease_matches, conversation_history, user_message):
    """Generate a response using the LLM"""
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Conversation history:\n{history}\n\nUser's latest message: {user_message}")
    ])
    
    # Set up the chain
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    
    # Generate response
    response = chain.invoke({
        "disease_matches": disease_matches,
        "history": conversation_history,
        "user_message": user_message
    })
    
    return response

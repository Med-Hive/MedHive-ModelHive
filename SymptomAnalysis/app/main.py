"""
Main FastAPI application for the Symptom Analysis Chatbot
"""
import logging
from fastapi import FastAPI
from app.api.routes import router
from app.services.database import initialize_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Symptom Analysis Chatbot")

# Include API routes
app.include_router(router)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import config
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
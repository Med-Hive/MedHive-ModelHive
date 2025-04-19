from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from app.core.config import get_settings
from app.routes import chat, health, root
import time
from app.core.logging_config import setup_logging
import psutil
import os

# Initialize logging
setup_logging(log_path=os.path.join(os.getcwd(), "logs"))

settings = get_settings()

app = FastAPI(
    title="MedHive Symptom Analysis API",
    description="AI-powered medical symptom analysis and diagnosis assistance",
    version="1.0.0"
)

# Include routers
app.include_router(root.router)
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Ensure logs directory exists
    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    pass
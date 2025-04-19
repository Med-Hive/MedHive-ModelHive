"""
Pydantic models for request and response validation
"""
from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    user: Optional[str] = None
    agent: Optional[str] = None

class ChatRequest(BaseModel):
    history: List[Message]

class ChatResponse(BaseModel):
    agent: str

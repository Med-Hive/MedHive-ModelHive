from pydantic import BaseModel, Field
from typing import List, Optional

class SymptomAnalysisRequest(BaseModel):
    symptoms: str = Field(..., description="Description of symptoms in natural language")
    additional_context: Optional[str] = Field(None, description="Additional patient context or medical history")
    
class TokenRequest(BaseModel):
    username: str
    password: str
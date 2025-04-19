from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, Dict

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "MedHive Symptom Analysis"
    
    # Groq Configuration
    GROQ_API_KEY: str
    GROQ_MODEL_NAME: str = "llama2-70b-4096"
    
    # Astra DB Configuration
    ASTRA_DB_APPLICATION_TOKEN: str
    ASTRA_DB_REGION: str
    ASTRA_DB_API_ENDPOINT: str
    ASTRA_DB_KEYSPACE: str = "disease_diagnosis"
    
    RATE_LIMIT: int = 1000  # Default rate limit per minute
    CORS_ORIGINS :str

    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_STRATEGY: str = "fixed-window"  # or "sliding-window"
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    RATE_LIMIT_MAX_REQUESTS: Dict[str, int] = {
        "default": 100,  # Default rate limit per window
        "/api/v1/chat/analyze": 20,  # Stricter limit for analysis endpoint
        "/api/v1/health": 200  # Higher limit for health checks
    }
    RATE_LIMIT_REDIS_URL: Optional[str] = None  # Optional Redis for distributed setup
    
    # Model Configuration
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.7
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
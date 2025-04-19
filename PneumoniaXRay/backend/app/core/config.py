from pydantic_settings import BaseSettings
import logging
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Pneumonia Detection API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Logging configuration
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Model configuration
    MODEL_PATH: str = "models/cnn_model.pkl"
    IMG_SIZE: int = 150  # Default image size for CNN model
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        env_file_encoding = "utf-8"
        json_schema_extra = {
            "example": {
                "MODEL_PATH": "models/cnn_model.pkl",
                "IMG_SIZE": 150,
                "LOG_LEVEL": "INFO"
            }
        }

settings = Settings()
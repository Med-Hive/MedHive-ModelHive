from pydantic_settings import BaseSettings
import logging
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Breast Cancer Detection API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Logging configuration
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Model configuration
    MODEL_PATH: str = "app/models/logistic_regression_model.pkl"
    METADATA_PATH: str = "app/models/model_metadata.pkl"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        env_file_encoding = "utf-8"
        json_schema_extra = {
            "example": {
                "MODEL_PATH": "app/models/model.pkl",
                "METADATA_PATH": "app/models/metadata.pkl",
                "LOG_LEVEL": "INFO"
            }
        }

settings = Settings()
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache

class MLOpsSettings(BaseSettings):
    # Federated Learning Configuration
    ENABLE_FEDERATED_LEARNING: bool = False
    FL_AGGREGATION_ROUNDS: int = 10
    FL_MIN_CLIENTS: int = 3
    FL_UPDATE_INTERVAL_HOURS: int = 24
    
    # Model Registry
    MODEL_REGISTRY_TYPE: str = "mlflow"  # or "wandb", "custom"
    MODEL_REGISTRY_URL: Optional[str] = None
    MODEL_VERSION_TAG: str = "production"
    
    # Performance Monitoring
    PERFORMANCE_METRICS_WINDOW: int = 1000  # Number of predictions to consider
    ACCURACY_THRESHOLD: float = 0.85
    LATENCY_THRESHOLD_MS: int = 500
    
    # Privacy Settings
    PRIVACY_TECHNIQUE: str = "differential_privacy"  # or "secure_aggregation", "encryption"
    EPSILON: float = 1.0  # Privacy budget for differential privacy
    
    # Monitoring
    ENABLE_DRIFT_DETECTION: bool = True
    DRIFT_DETECTION_WINDOW: int = 1000
    SIGNIFICANT_DRIFT_THRESHOLD: float = 0.1
    
    # Client Configuration
    FL_CLIENT_UPDATE_BATCH_SIZE: int = 32
    FL_CLIENT_LOCAL_EPOCHS: int = 1
    FL_CLIENT_LEARNING_RATE: float = 0.001
    
    # Model Versioning
    ENABLE_MODEL_VERSIONING: bool = True
    AUTO_ROLLBACK_ON_DEGRADATION: bool = True
    ROLLBACK_WINDOW_HOURS: int = 24
    
    # A/B Testing
    ENABLE_AB_TESTING: bool = False
    AB_TEST_TRAFFIC_SPLIT: List[float] = [0.9, 0.1]  # [production, experiment]
    MIN_SAMPLES_FOR_SIGNIFICANCE: int = 1000
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_mlops_settings() -> MLOpsSettings:
    return MLOpsSettings()

def is_federated_learning_ready() -> bool:
    """Check if the system is ready for federated learning"""
    settings = get_mlops_settings()
    return (
        settings.ENABLE_FEDERATED_LEARNING
        and settings.FL_MIN_CLIENTS > 0
        and settings.MODEL_REGISTRY_URL is not None
    )

def get_current_model_performance() -> dict:
    """Get current model performance metrics"""
    settings = get_mlops_settings()
    # Placeholder for actual implementation
    return {
        "accuracy": 0.0,
        "latency": 0.0,
        "drift_detected": False,
        "privacy_budget_remaining": settings.EPSILON
    }

def should_trigger_retraining() -> bool:
    """Determine if model retraining should be triggered"""
    settings = get_mlops_settings()
    performance = get_current_model_performance()
    
    return (
        performance["accuracy"] < settings.ACCURACY_THRESHOLD
        or performance["latency"] > settings.LATENCY_THRESHOLD_MS
        or performance["drift_detected"]
    )
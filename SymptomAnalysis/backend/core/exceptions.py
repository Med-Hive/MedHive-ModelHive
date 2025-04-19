from typing import Optional
from fastapi import HTTPException

class ModelServiceError(Exception):
    """Base exception for model service errors"""
    def __init__(self, message: str, error_code: str, status_code: int = 500):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)

class ModelInferenceError(ModelServiceError):
    """Raised when model inference fails"""
    def __init__(self, message: str, model_name: str):
        super().__init__(
            message=f"Model inference failed: {message}",
            error_code="MODEL_INFERENCE_ERROR",
            status_code=500
        )
        self.model_name = model_name

class DriftDetectionError(ModelServiceError):
    """Raised when drift detection fails"""
    def __init__(self, message: str, drift_score: Optional[float] = None):
        super().__init__(
            message=f"Drift detection error: {message}",
            error_code="DRIFT_DETECTION_ERROR",
            status_code=500
        )
        self.drift_score = drift_score

class PerformanceDegradationError(ModelServiceError):
    """Raised when model performance degrades significantly"""
    def __init__(self, message: str, metrics: dict):
        super().__init__(
            message=f"Performance degradation detected: {message}",
            error_code="PERFORMANCE_DEGRADATION",
            status_code=500
        )
        self.metrics = metrics

class MLOpsTrackingError(ModelServiceError):
    """Raised when MLOps tracking operations fail"""
    def __init__(self, message: str, operation: str):
        super().__init__(
            message=f"MLOps tracking failed for operation '{operation}': {message}",
            error_code="MLOPS_TRACKING_ERROR",
            status_code=500
        )
        self.operation = operation

class FederatedLearningError(ModelServiceError):
    """Raised when federated learning operations fail"""
    def __init__(self, message: str, round_number: Optional[int] = None):
        super().__init__(
            message=f"Federated learning error: {message}",
            error_code="FL_ERROR",
            status_code=500
        )
        self.round_number = round_number

def handle_model_service_error(error: ModelServiceError) -> HTTPException:
    """Convert ModelServiceError to FastAPI HTTPException"""
    return HTTPException(
        status_code=error.status_code,
        detail={
            "message": error.message,
            "error_code": error.error_code,
            "error_type": error.__class__.__name__
        }
    )
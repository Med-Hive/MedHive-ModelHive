from fastapi import HTTPException

class APIException(HTTPException):
    """Base exception class for API errors"""
    def __init__(self, status_code: int, error_code: str, detail: str):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code

class ModelLoadError(APIException):
    """Exception raised when model fails to load"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=500,
            error_code="MODEL_LOAD_ERROR",
            detail=detail
        )

class PredictionError(APIException):
    """Exception raised when prediction fails"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=500,
            error_code="PREDICTION_ERROR",
            detail=detail
        )

class FeatureError(APIException):
    """Exception raised when feature processing fails"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=400,
            error_code="FEATURE_ERROR",
            detail=detail
        ) 
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Disease(BaseModel):
    name: str
    confidence: float
    symptoms_matched: List[str]
    recommendation: Optional[str] = None

class SymptomAnalysisResponse(BaseModel):
    possible_diseases: List[Disease]
    analysis_summary: str
    severity_level: Optional[str] = None
    seek_immediate_care: bool = False
    prediction_id: Optional[str] = None  # Added to track feedback

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ErrorDetail(BaseModel):
    message: str
    error_code: str
    error_type: str
    additional_info: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    message: str
    error_code: str
    error_type: str
    request_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class ChatResponse(BaseModel):
    text: str
    confidence: float
    symptoms: List[str]
    possible_conditions: List[Dict[str, Any]]
    severity: str
    request_id: str
    model_version: str
    latency_ms: float

class ModelMetrics(BaseModel):
    total_requests: int
    total_errors: int
    error_rate: float
    average_latency_ms: float
    max_drift_score: float
    metrics_count: int
    time_range: Dict[str, Optional[str]]

class ModelHealth(BaseModel):
    status: str = Field(description="Overall model health status: healthy, degraded, or unhealthy")
    timestamp: str
    model_version: str
    monitoring_metrics: ModelMetrics
    resource_metrics: Dict[str, float] = Field(description="System resource utilization metrics")
    federated_learning: Dict[str, Any] = Field(description="Federated learning status and metrics")
    drift_detection: Dict[str, Any] = Field(description="Model drift detection status and metrics")

class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: Optional[str] = None
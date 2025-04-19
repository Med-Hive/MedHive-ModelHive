from fastapi import APIRouter, HTTPException
from typing import Dict
from datetime import datetime
import psutil
import os
from app.core.config import get_settings
from app.services.chat_service import ChatService
from app.models.responses import ModelHealth, ModelMetrics
from app.core.context import RequestContext

router = APIRouter()
settings = get_settings()
chat_service = ChatService()

@router.get("/", response_model=dict)
async def health_check():
    """Basic health check endpoint"""
    with RequestContext() as ctx:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": ctx.request_id
        }

@router.get("/model", response_model=ModelHealth)
async def model_health():
    """Get detailed model health and performance metrics"""
    with RequestContext() as ctx:
        try:
            # Get model metrics for the last 24 hours
            metrics = await chat_service.mlops_tracker.get_model_performance(
                model_version_id=chat_service.current_model_version,
                time_window="24h"
            )
            
            # Get current drift detection status
            drift_metrics = chat_service.model_monitor.get_performance_metrics()
            
            # Get system resource metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            return ModelHealth(
                status="healthy" if drift_metrics.get("drift_score", 0) < 0.3 else "degraded",
                timestamp=datetime.utcnow().isoformat(),
                model_version=chat_service.current_model_version,
                monitoring_metrics=ModelMetrics(
                    total_requests=metrics.get("total_requests", 0),
                    total_errors=metrics.get("total_errors", 0),
                    error_rate=metrics.get("error_rate", 0),
                    average_latency_ms=metrics.get("average_latency_ms", 0),
                    max_drift_score=metrics.get("max_drift_score", 0),
                    metrics_count=metrics.get("metrics_count", 0),
                    time_range=metrics.get("time_range", {"start": None, "end": None})
                ),
                resource_metrics={
                    "cpu_percent": cpu_percent,
                    "memory_used_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024 * 1024 * 1024)
                },
                federated_learning={
                    "enabled": os.getenv("ENABLE_FEDERATED_LEARNING", "false").lower() == "true",
                    "last_aggregation": None,  # To be implemented with federated learning
                    "participating_clients": 0
                },
                drift_detection={
                    "current_drift_score": drift_metrics.get("drift_score", 0),
                    "last_check": drift_metrics.get("last_check", None),
                    "status": "normal" if drift_metrics.get("drift_score", 0) < 0.3 else "warning"
                }
            )
        except Exception as e:
            RequestContext.set_metadata("error", str(e))
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Error getting model health metrics",
                    "error": str(e),
                    "request_id": ctx.request_id
                }
            )
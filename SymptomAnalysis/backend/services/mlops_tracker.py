from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import httpx
from app.core.config import get_settings
from app.core.mlops_config import get_mlops_settings
from app.core.exceptions import MLOpsTrackingError
from app.core.context import RequestContext
from app.core.logging_config import get_mlops_logger
import logging
import json
import uuid

logger = get_mlops_logger()
settings = get_settings()
mlops_settings = get_mlops_settings()

class MLOpsTracker:
    def __init__(self):
        self.supabase_url = settings.SUPABASE_URL
        self.supabase_key = settings.SUPABASE_KEY

    async def track_prediction(
        self,
        input_text: str,
        input_embedding: List[float],
        prediction: Dict[str, Any],
        confidence: float,
        latency_ms: float,
        model_version_id: str
    ) -> None:
        """Track a model prediction in Supabase"""
        try:
            request_id = RequestContext.get_request_id()
            user_id = RequestContext.get_user_id()
            
            prediction_data = {
                "id": str(uuid.uuid4()),
                "model_version_id": model_version_id,
                "input_text": input_text,
                "input_embedding": input_embedding,
                "prediction": prediction,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "created_at": datetime.utcnow().isoformat(),
                "request_id": request_id,
                "user_id": user_id,
                "context": RequestContext.get_all_metadata()
            }

            logger.info(
                "Tracking prediction",
                extra={
                    "request_id": request_id,
                    "prediction_id": prediction_data["id"],
                    "model_version": model_version_id,
                    "latency_ms": latency_ms
                }
            )

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.supabase_url}/rest/v1/model_predictions",
                    headers={
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                        "X-Request-ID": request_id
                    },
                    json=prediction_data,
                    timeout=10.0
                )
                response.raise_for_status()
                
        except httpx.TimeoutException as e:
            logger.error(
                "Timeout while tracking prediction",
                extra={
                    "request_id": RequestContext.get_request_id(),
                    "error": str(e)
                }
            )
            raise MLOpsTrackingError(
                message="Timeout while tracking prediction",
                operation="track_prediction"
            ) from e
        except Exception as e:
            logger.error(
                "Error tracking prediction",
                extra={
                    "request_id": RequestContext.get_request_id(),
                    "error": str(e)
                },
                exc_info=True
            )
            raise MLOpsTrackingError(
                message=str(e),
                operation="track_prediction"
            ) from e

    async def record_feedback(
        self,
        prediction_id: str,
        user_id: str,
        correct_diagnosis: Optional[str],
        feedback_text: Optional[str],
        severity_reported: str
    ) -> None:
        """Record user feedback for a prediction"""
        try:
            request_id = RequestContext.get_request_id()
            
            feedback_data = {
                "id": str(uuid.uuid4()),
                "prediction_id": prediction_id,
                "user_id": user_id,
                "correct_diagnosis": correct_diagnosis,
                "feedback_text": feedback_text,
                "severity_reported": severity_reported,
                "created_at": datetime.utcnow().isoformat(),
                "request_id": request_id,
                "context": RequestContext.get_all_metadata()
            }

            logger.info(
                "Recording feedback",
                extra={
                    "request_id": request_id,
                    "feedback_id": feedback_data["id"],
                    "prediction_id": prediction_id
                }
            )

            async with httpx.AsyncClient() as client:
                # Record feedback
                response = await client.post(
                    f"{self.supabase_url}/rest/v1/prediction_feedback",
                    headers={
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                        "X-Request-ID": request_id
                    },
                    json=feedback_data,
                    timeout=10.0
                )
                response.raise_for_status()

                # Update the prediction to mark feedback as received
                await client.patch(
                    f"{self.supabase_url}/rest/v1/model_predictions",
                    headers={
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                        "X-Request-ID": request_id
                    },
                    params={"id": f"eq.{prediction_id}"},
                    json={"feedback_received": True},
                    timeout=10.0
                )

        except httpx.TimeoutException as e:
            logger.error(
                "Timeout while recording feedback",
                extra={
                    "request_id": RequestContext.get_request_id(),
                    "error": str(e)
                }
            )
            raise MLOpsTrackingError(
                message="Timeout while recording feedback",
                operation="record_feedback"
            ) from e
        except Exception as e:
            logger.error(
                "Error recording feedback",
                extra={
                    "request_id": RequestContext.get_request_id(),
                    "error": str(e)
                },
                exc_info=True
            )
            raise MLOpsTrackingError(
                message=str(e),
                operation="record_feedback"
            ) from e

    async def update_model_metrics(
        self,
        model_version_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Update performance metrics for a model version"""
        try:
            request_id = RequestContext.get_request_id()
            
            metrics_data = {
                "id": str(uuid.uuid4()),
                "model_version_id": model_version_id,
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id,
                "context": RequestContext.get_all_metadata(),
                **metrics
            }

            logger.info(
                "Updating model metrics",
                extra={
                    "request_id": request_id,
                    "model_version": model_version_id,
                    "metrics": metrics
                }
            )

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.supabase_url}/rest/v1/performance_metrics",
                    headers={
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                        "X-Request-ID": request_id
                    },
                    json=metrics_data,
                    timeout=10.0
                )
                response.raise_for_status()

        except httpx.TimeoutException as e:
            logger.error(
                "Timeout while updating metrics",
                extra={
                    "request_id": RequestContext.get_request_id(),
                    "error": str(e)
                }
            )
            raise MLOpsTrackingError(
                message="Timeout while updating metrics",
                operation="update_model_metrics"
            ) from e
        except Exception as e:
            logger.error(
                "Error updating model metrics",
                extra={
                    "request_id": RequestContext.get_request_id(),
                    "error": str(e)
                },
                exc_info=True
            )
            raise MLOpsTrackingError(
                message=str(e),
                operation="update_model_metrics"
            ) from e

    async def get_model_performance(
        self,
        model_version_id: str,
        time_window: str = "24h"
    ) -> Dict[str, Any]:
        """Get model performance metrics for a specific time window"""
        try:
            request_id = RequestContext.get_request_id()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.supabase_url}/rest/v1/performance_metrics",
                    headers={
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                        "X-Request-ID": request_id
                    },
                    params={
                        "model_version_id": f"eq.{model_version_id}",
                        "timestamp": f"gte.{self._get_time_window(time_window)}",
                        "order": "timestamp.desc"
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                metrics = self._aggregate_metrics(response.json())
                
                logger.info(
                    "Retrieved model performance metrics",
                    extra={
                        "request_id": request_id,
                        "model_version": model_version_id,
                        "time_window": time_window,
                        "metrics_count": metrics.get("metrics_count", 0)
                    }
                )
                
                return metrics

        except httpx.TimeoutException as e:
            logger.error(
                "Timeout while fetching performance metrics",
                extra={
                    "request_id": RequestContext.get_request_id(),
                    "error": str(e)
                }
            )
            raise MLOpsTrackingError(
                message="Timeout while fetching performance metrics",
                operation="get_model_performance"
            ) from e
        except Exception as e:
            logger.error(
                "Error fetching model performance",
                extra={
                    "request_id": RequestContext.get_request_id(),
                    "error": str(e)
                },
                exc_info=True
            )
            raise MLOpsTrackingError(
                message=str(e),
                operation="get_model_performance"
            ) from e
            
    def _get_time_window(self, window: str) -> str:
        """Convert time window string to ISO timestamp"""
        now = datetime.utcnow()
        hours = {
            "1h": 1,
            "6h": 6,
            "12h": 12,
            "24h": 24,
            "7d": 24 * 7,
            "30d": 24 * 30
        }.get(window, 24)
        
        window_time = now - timedelta(hours=hours)
        return window_time.isoformat()

    def _aggregate_metrics(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics for the time period"""
        if not metrics:
            return {}

        total_requests = sum(m.get("request_count", 0) for m in metrics)
        total_errors = sum(m.get("error_count", 0) for m in metrics)
        avg_latency = sum(m.get("average_latency_ms", 0) for m in metrics) / len(metrics)
        max_drift = max(m.get("drift_score", 0) for m in metrics)

        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_requests) if total_requests > 0 else 0,
            "average_latency_ms": avg_latency,
            "max_drift_score": max_drift,
            "metrics_count": len(metrics),
            "time_range": {
                "start": metrics[-1].get("timestamp"),
                "end": metrics[0].get("timestamp")
            }
        }
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from app.core.metrics import MetricsTracker
from app.core.logging_config import get_logger
from dataclasses import dataclass
import threading
import time

logger = get_logger(__name__)

@dataclass
class ModelMetrics:
    latency_p95: float
    latency_p99: float
    error_rate: float
    request_count: int
    last_error: Optional[str]
    last_updated: datetime

class ModelMonitor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.metrics_window = timedelta(minutes=5)
        self.latencies: List[float] = []
        self.errors: List[str] = []
        self.request_times: List[datetime] = []
        self.last_error: Optional[str] = None
        self.last_updated = datetime.now()
        # Drift detection additions
        self.predictions: List[Dict] = []
        self.reference_distribution: Optional[Dict] = None
        self.drift_window_size = 1000
        self.drift_threshold = 0.1

    def record_prediction(self, latency: float, prediction: Dict, error: Optional[str] = None):
        """Record a model prediction with its latency, prediction data, and any error"""
        now = datetime.now()
        
        with self._lock:
            # Clean old data
            self._clean_old_data(now)
            
            # Record new data
            self.latencies.append(latency)
            self.request_times.append(now)
            self.predictions.append(prediction)
            
            if error:
                self.errors.append(error)
                self.last_error = error
                MetricsTracker.track_model_error("symptom_analyzer", error)
            
            self.last_updated = now
            
            # Check for drift
            self._check_drift()

    def get_current_metrics(self) -> ModelMetrics:
        """Get current model metrics"""
        with self._lock:
            if not self.latencies:
                return ModelMetrics(
                    latency_p95=0.0,
                    latency_p99=0.0,
                    error_rate=0.0,
                    request_count=0,
                    last_error=None,
                    last_updated=self.last_updated
                )

            latencies = np.array(self.latencies)
            total_requests = len(self.request_times)
            error_rate = len(self.errors) / total_requests if total_requests > 0 else 0.0

            return ModelMetrics(
                latency_p95=np.percentile(latencies, 95),
                latency_p99=np.percentile(latencies, 99),
                error_rate=error_rate,
                request_count=total_requests,
                last_error=self.last_error,
                last_updated=self.last_updated
            )

    def _clean_old_data(self, current_time: datetime):
        """Remove data older than the metrics window"""
        cutoff_time = current_time - self.metrics_window
        
        # Find the cutoff index
        cutoff_idx = 0
        for i, req_time in enumerate(self.request_times):
            if req_time >= cutoff_time:
                cutoff_idx = i
                break
        
        # Remove old data
        if cutoff_idx > 0:
            self.latencies = self.latencies[cutoff_idx:]
            self.request_times = self.request_times[cutoff_idx:]
            self.errors = self.errors[cutoff_idx:]

    def check_model_health(self) -> Dict[str, any]:
        """Check overall model health based on current metrics"""
        metrics = self.get_current_metrics()
        
        health_status = "healthy"
        issues = []
        
        # Check latency thresholds
        if metrics.latency_p95 > 2.0:  # More than 2 seconds for p95
            issues.append("High latency detected (p95 > 2s)")
            health_status = "degraded"
        
        # Check error rate
        if metrics.error_rate > 0.05:  # More than 5% error rate
            issues.append(f"High error rate detected ({metrics.error_rate:.1%})")
            health_status = "degraded"
        
        # Check request volume
        if metrics.request_count == 0:
            issues.append("No requests in current window")
            health_status = "unknown"
        
        return {
            "status": health_status,
            "issues": issues,
            "metrics": {
                "latency_p95": f"{metrics.latency_p95:.3f}s",
                "latency_p99": f"{metrics.latency_p99:.3f}s",
                "error_rate": f"{metrics.error_rate:.1%}",
                "request_count": metrics.request_count,
                "last_error": metrics.last_error,
                "last_updated": metrics.last_updated.isoformat()
            }
        }

    def set_reference_distribution(self, reference_data: List[Dict]):
        """Set the reference distribution for drift detection"""
        if len(reference_data) > 0:
            self.reference_distribution = self._calculate_distribution(reference_data)
            logger.info("Reference distribution updated for drift detection")

    def _calculate_distribution(self, predictions: List[Dict]) -> Dict:
        """Calculate the distribution of predictions"""
        # Convert predictions to a format suitable for distribution analysis
        # For example, counting frequency of each predicted disease
        distribution = {}
        for pred in predictions:
            disease = pred.get("predicted_disease", "unknown")
            distribution[disease] = distribution.get(disease, 0) + 1
        
        # Normalize to get probabilities
        total = sum(distribution.values())
        return {k: v/total for k, v in distribution.items()}

    def _check_drift(self):
        """Check for drift in recent predictions"""
        if len(self.predictions) < self.drift_window_size or not self.reference_distribution:
            return

        # Calculate current distribution
        recent_predictions = self.predictions[-self.drift_window_size:]
        current_distribution = self._calculate_distribution(recent_predictions)

        # Calculate KL divergence as drift metric
        drift_score = self._calculate_kl_divergence(self.reference_distribution, current_distribution)

        if drift_score > self.drift_threshold:
            logger.warning(f"Significant drift detected! Score: {drift_score:.4f}")
            MetricsTracker.track_drift_alert("symptom_analyzer", drift_score)

    def _calculate_kl_divergence(self, p: Dict, q: Dict) -> float:
        """Calculate KL divergence between two distributions"""
        # Ensure both distributions have the same keys
        all_keys = set(p.keys()) | set(q.keys())
        kl_div = 0.0
        
        for key in all_keys:
            p_val = p.get(key, 1e-10)  # Small epsilon to avoid division by zero
            q_val = q.get(key, 1e-10)
            
            if p_val > 0:
                kl_div += p_val * np.log(p_val / q_val)
        
        return kl_div
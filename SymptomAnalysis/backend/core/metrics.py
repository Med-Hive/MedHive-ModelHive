from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['endpoint', 'method', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['endpoint', 'method'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Model metrics
MODEL_PREDICTION_TIME = Histogram(
    'model_prediction_duration_seconds',
    'Time spent on model predictions',
    ['model_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

MODEL_ERRORS = Counter(
    'model_errors_total',
    'Total number of model errors',
    ['model_name', 'error_type']
)

# System metrics
SYSTEM_CPU = Gauge('system_cpu_usage', 'Current CPU usage percentage')
SYSTEM_MEMORY = Gauge('system_memory_usage_bytes', 'Current memory usage in bytes')

class MetricsTracker:
    @staticmethod
    def track_request(endpoint: str, method: str, status: int, start_time: float) -> None:
        duration = time.time() - start_time
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)

    @staticmethod
    def track_model_prediction(model_name: str, duration: float) -> None:
        MODEL_PREDICTION_TIME.labels(model_name=model_name).observe(duration)

    @staticmethod
    def track_model_error(model_name: str, error_type: str) -> None:
        MODEL_ERRORS.labels(model_name=model_name, error_type=error_type).inc()
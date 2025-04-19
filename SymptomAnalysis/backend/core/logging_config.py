import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }

        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "duration_ms"):
            log_obj["duration_ms"] = record.duration_ms

        if hasattr(record, "model_version"):
            log_obj["model_version"] = record.model_version

        return json.dumps(log_obj)

def setup_logging(log_path: str = "logs") -> None:
    """Setup application logging with file and console handlers"""
    # Create logs directory if it doesn't exist
    Path(log_path).mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_path, "app.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    error_file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_path, "error.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )

    # Set levels
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    error_file_handler.setLevel(logging.ERROR)

    # Create formatters
    json_formatter = JSONFormatter()
    
    # Set formatters
    console_handler.setFormatter(json_formatter)
    file_handler.setFormatter(json_formatter)
    error_file_handler.setFormatter(json_formatter)

    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_file_handler)

    # Create separate logger for model predictions
    model_logger = logging.getLogger("model_predictions")
    model_logger.setLevel(logging.INFO)
    model_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_path, "model_predictions.log"),
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10
    )
    model_handler.setFormatter(json_formatter)
    model_logger.addHandler(model_handler)

    # Create separate logger for MLOps metrics
    mlops_logger = logging.getLogger("mlops")
    mlops_logger.setLevel(logging.INFO)
    mlops_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_path, "mlops.log"),
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=7
    )
    mlops_handler.setFormatter(json_formatter)
    mlops_logger.addHandler(mlops_handler)

def get_model_logger() -> logging.Logger:
    """Get logger for model predictions"""
    return logging.getLogger("model_predictions")

def get_mlops_logger() -> logging.Logger:
    """Get logger for MLOps metrics"""
    return logging.getLogger("mlops")
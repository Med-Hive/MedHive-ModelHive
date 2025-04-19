import contextvars
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

# Context variables
request_id_ctx = contextvars.ContextVar("request_id", default=None)
user_id_ctx = contextvars.ContextVar("user_id", default=None)
start_time_ctx = contextvars.ContextVar("start_time", default=None)
metadata_ctx = contextvars.ContextVar("metadata", default={})

class RequestContext:
    """Context manager for request-scoped data"""
    
    def __init__(self, request_id: Optional[str] = None, user_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.user_id = user_id
        self.start_time = datetime.utcnow()
        self.tokens = []

    def __enter__(self):
        # Set context variables
        self.tokens.extend([
            request_id_ctx.set(self.request_id),
            user_id_ctx.set(self.user_id),
            start_time_ctx.set(self.start_time),
            metadata_ctx.set({})
        ])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset context variables
        for token in self.tokens:
            contextvars.ContextVar.reset(token)

    @staticmethod
    def get_request_id() -> Optional[str]:
        """Get current request ID"""
        return request_id_ctx.get()

    @staticmethod
    def get_user_id() -> Optional[str]:
        """Get current user ID"""
        return user_id_ctx.get()

    @staticmethod
    def get_duration_ms() -> float:
        """Get request duration in milliseconds"""
        start_time = start_time_ctx.get()
        if start_time:
            return (datetime.utcnow() - start_time).total_seconds() * 1000
        return 0.0

    @staticmethod
    def set_metadata(key: str, value: Any) -> None:
        """Set metadata for the current request"""
        metadata = metadata_ctx.get()
        metadata[key] = value
        metadata_ctx.set(metadata)

    @staticmethod
    def get_metadata(key: str, default: Any = None) -> Any:
        """Get metadata for the current request"""
        return metadata_ctx.get().get(key, default)

    @staticmethod
    def get_all_metadata() -> Dict[str, Any]:
        """Get all metadata for the current request"""
        return metadata_ctx.get().copy()

    @staticmethod
    def get_context_dict() -> Dict[str, Any]:
        """Get all context information as a dictionary"""
        return {
            "request_id": request_id_ctx.get(),
            "user_id": user_id_ctx.get(),
            "duration_ms": RequestContext.get_duration_ms(),
            "metadata": metadata_ctx.get()
        }
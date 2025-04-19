from fastapi import Request
from fastapi.responses import JSONResponse
from app.core.exceptions import ModelServiceError, handle_model_service_error
from app.core.logging_config import get_model_logger, get_mlops_logger
from app.core.context import RequestContext
import logging
import uuid
from typing import Callable
import time
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp
from starlette.responses import Response
from app.core.metrics import MetricsTracker
from app.core.logging_config import get_request_logger

logger = logging.getLogger(__name__)
model_logger = get_model_logger()
mlops_logger = get_mlops_logger()
request_logger = get_request_logger()

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        """Process request with request context and logging"""
        # Get user ID from request state if available
        user_id = getattr(request.state, "user_id", None)
        
        with RequestContext(user_id=user_id) as ctx:
            request_id = ctx.request_id
            
            # Store request_id in request state for other middlewares
            request.state.request_id = request_id
            
            # Add initial metadata
            ctx.set_metadata("path", request.url.path)
            ctx.set_metadata("method", request.method)
            ctx.set_metadata("client_ip", request.client.host if request.client else None)
            
            # Log request start
            logger.info(
                "Request started",
                extra=ctx.get_context_dict()
            )

            try:
                response = await call_next(request)
                
                # Add response metadata
                ctx.set_metadata("status_code", response.status_code)
                ctx.set_metadata("duration_ms", ctx.get_duration_ms())
                
                # Log request completion
                logger.info(
                    "Request completed",
                    extra=ctx.get_context_dict()
                )

                # Add request ID to response headers
                response.headers["X-Request-ID"] = request_id
                return response

            except Exception as e:
                # Add error metadata
                ctx.set_metadata("error", str(e))
                ctx.set_metadata("error_type", e.__class__.__name__)
                
                # Log error
                logger.error(
                    "Request failed",
                    extra=ctx.get_context_dict(),
                    exc_info=True
                )
                raise

class ModelErrorMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except ModelServiceError as e:
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            logger.error(
                f"Model service error: {str(e)}",
                extra={
                    "request_id": request_id,
                    "error_code": e.error_code,
                    "error_type": e.__class__.__name__,
                    "context": RequestContext.get_all_metadata()
                },
                exc_info=True
            )
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "detail": {
                        "message": e.message,
                        "error_code": e.error_code,
                        "error_type": e.__class__.__name__,
                        "request_id": request_id
                    }
                }
            )
        except Exception as e:
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            logger.error(
                f"Unhandled error: {str(e)}",
                extra={
                    "request_id": request_id,
                    "context": RequestContext.get_all_metadata()
                },
                exc_info=True
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": {
                        "message": "An unexpected error occurred",
                        "error_code": "INTERNAL_SERVER_ERROR",
                        "error_type": "UnhandledException",
                        "request_id": request_id
                    }
                }
            )

class RequestTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.time()
        
        with RequestContext() as ctx:
            ctx.set_request_id(request_id)
            ctx.set_user_id(request.headers.get("X-User-ID"))
            ctx.set_metadata("ip_address", request.client.host)
            ctx.set_metadata("user_agent", request.headers.get("User-Agent"))
            
            try:
                response = await call_next(request)
                
                # Track request metrics
                MetricsTracker.track_request(
                    endpoint=request.url.path,
                    method=request.method,
                    status=response.status_code,
                    start_time=start_time
                )
                
                # Add request ID to response headers
                response.headers["X-Request-ID"] = request_id
                
                # Log request completion
                request_logger.info(
                    f"Request completed: {request.method} {request.url.path}",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration": time.time() - start_time
                    }
                )
                
                return response
                
            except Exception as e:
                request_logger.error(
                    f"Request failed: {str(e)}",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "error": str(e)
                    },
                    exc_info=True
                )
                raise

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limit_per_minute: int = 60):
        super().__init__(app)
        self.rate_limit = rate_limit_per_minute
        self.requests = {}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean up old requests
        self.cleanup_old_requests(current_time)
        
        # Check if client has exceeded rate limit
        if self.is_rate_limited(client_ip, current_time):
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": "60"}
            )
        
        # Track request
        self.track_request(client_ip, current_time)
        
        return await call_next(request)
    
    def cleanup_old_requests(self, current_time: float) -> None:
        cutoff_time = current_time - 60  # 1 minute window
        self.requests = {
            ip: [ts for ts in timestamps if ts > cutoff_time]
            for ip, timestamps in self.requests.items()
        }
        # Remove empty entries
        self.requests = {
            ip: timestamps
            for ip, timestamps in self.requests.items()
            if timestamps
        }
    
    def is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        if client_ip not in self.requests:
            return False
        
        # Count requests in the last minute
        recent_requests = len([
            ts for ts in self.requests[client_ip]
            if ts > current_time - 60
        ])
        
        return recent_requests >= self.rate_limit
    
    def track_request(self, client_ip: str, current_time: float) -> None:
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip].append(current_time)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response
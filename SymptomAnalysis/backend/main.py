from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from app.core.config import get_settings
from app.routes import chat, health, root
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, make_asgi_app
import time
from app.core.metrics import REQUEST_COUNT, REQUEST_LATENCY, SYSTEM_CPU, SYSTEM_MEMORY
from app.core.middleware import RequestLoggingMiddleware, ModelErrorMiddleware, RequestTracingMiddleware, RateLimitMiddleware, SecurityHeadersMiddleware
from app.core.logging_config import setup_logging
import psutil
import os

# Initialize logging
setup_logging(log_path=os.path.join(os.getcwd(), "logs"))

settings = get_settings()

app = FastAPI(
    title="MedHive Symptom Analysis API",
    description="AI-powered medical symptom analysis and diagnosis assistance",
    version="1.0.0"
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Add middleware in the correct order
# 1. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 2. Security headers
app.add_middleware(SecurityHeadersMiddleware)
# 3. Rate limiting (first to reject excess requests early)
app.add_middleware(RateLimitMiddleware, rate_limit_per_minute=settings.RATE_LIMIT)
# 4. Request tracing
app.add_middleware(RequestTracingMiddleware)
# 5. Model error handling
app.add_middleware(ModelErrorMiddleware)
# 6. Request logging for all requests
app.add_middleware(RequestLoggingMiddleware)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to track request metrics"""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Record request count and latency
    REQUEST_COUNT.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(
        endpoint=request.url.path,
        method=request.method
    ).observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    """Endpoint for Prometheus metrics"""
    # Update system metrics
    SYSTEM_CPU.set(psutil.cpu_percent())
    SYSTEM_MEMORY.set(psutil.virtual_memory().used)
    
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Include routers
app.include_router(root.router)
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Ensure logs directory exists
    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    pass
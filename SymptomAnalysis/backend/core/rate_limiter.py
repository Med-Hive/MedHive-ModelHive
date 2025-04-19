from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from app.core.config import get_settings
from datetime import datetime, timezone
import time
from typing import Dict, Optional, Tuple
import redis
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

class RateLimiter:
    """Rate limiter implementation supporting both in-memory and Redis storage"""
    def __init__(self):
        self.strategy = settings.RATE_LIMIT_STRATEGY
        self.window_seconds = settings.RATE_LIMIT_WINDOW_SECONDS
        self.max_requests = settings.RATE_LIMIT_MAX_REQUESTS
        self.redis_url = settings.RATE_LIMIT_REDIS_URL
        
        # Initialize storage
        if self.redis_url:
            self.storage = redis.from_url(self.redis_url)
            logger.info("Using Redis for rate limiting")
        else:
            self._in_memory_storage: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
            self._last_cleanup = time.time()
            logger.info("Using in-memory storage for rate limiting")

    async def is_rate_limited(self, identifier: str, path: str) -> Tuple[bool, Dict[str, int]]:
        """Check if request should be rate limited"""
        try:
            current_window = int(time.time() / self.window_seconds)
            max_requests = self.max_requests.get(path, self.max_requests["default"])
            
            if self.redis_url:
                return await self._check_redis_limit(identifier, current_window, max_requests)
            else:
                return self._check_memory_limit(identifier, current_window, max_requests)
                
        except Exception as e:
            logger.error(f"Rate limiting error: {str(e)}")
            return False, {"remaining": 0, "reset": 0}  # Fail open on errors

    async def _check_redis_limit(
        self, identifier: str, current_window: int, max_requests: int
    ) -> Tuple[bool, Dict[str, int]]:
        """Check rate limit using Redis"""
        key = f"rl:{identifier}:{current_window}"
        pipe = self.storage.pipeline()
        
        try:
            # Increment counter and set expiry
            count = pipe.incr(key)
            pipe.expire(key, self.window_seconds)
            pipe.execute()
            
            remaining = max(0, max_requests - count)
            reset = (current_window + 1) * self.window_seconds
            
            return count > max_requests, {
                "remaining": remaining,
                "reset": reset
            }
        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiter: {str(e)}")
            return False, {"remaining": 0, "reset": 0}

    def _check_memory_limit(
        self, identifier: str, current_window: int, max_requests: int
    ) -> Tuple[bool, Dict[str, int]]:
        """Check rate limit using in-memory storage"""
        # Cleanup old windows periodically
        if time.time() - self._last_cleanup > 60:  # Cleanup every minute
            self._cleanup_old_windows()

        # Increment counter for current window
        self._in_memory_storage[identifier][current_window] += 1
        count = self._in_memory_storage[identifier][current_window]
        
        remaining = max(0, max_requests - count)
        reset = (current_window + 1) * self.window_seconds
        
        return count > max_requests, {
            "remaining": remaining,
            "reset": reset
        }

    def _cleanup_old_windows(self) -> None:
        """Remove expired windows from memory"""
        current_window = int(time.time() / self.window_seconds)
        for identifier in list(self._in_memory_storage.keys()):
            self._in_memory_storage[identifier] = {
                window: count
                for window, count in self._in_memory_storage[identifier].items()
                if window >= current_window - 1
            }
            if not self._in_memory_storage[identifier]:
                del self._in_memory_storage[identifier]
        self._last_cleanup = time.time()

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.limiter = RateLimiter()
        logger.info("Rate limiting middleware initialized")

    async def dispatch(self, request: Request, call_next):
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        # Get client identifier (IP or user ID if authenticated)
        identifier = self._get_client_identifier(request)
        path = request.url.path

        # Check rate limit
        is_limited, limit_info = await self.limiter.is_rate_limited(identifier, path)
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(limit_info["reset"])
        
        if is_limited:
            logger.warning(f"Rate limit exceeded for {identifier} on {path}")
            raise HTTPException(
                status_code=429,
                detail={
                    "message": "Too many requests",
                    "reset": limit_info["reset"]
                }
            )
        
        return response

    def _get_client_identifier(self, request: Request) -> str:
        """Get unique identifier for rate limiting"""
        # If user is authenticated, use user ID
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"
            
        # Otherwise use IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0]}"
        return f"ip:{request.client.host if request.client else 'unknown'}"
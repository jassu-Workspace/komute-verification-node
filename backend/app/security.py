import time
from collections import defaultdict
from typing import Optional
from fastapi import Header, HTTPException, Request, status
from app.config import settings


# In-memory sliding window rate limiter
_request_records: dict[str, list[float]] = defaultdict(list)


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> None:
    """Validate API key header if authentication is enabled."""
    if not settings.enable_api_key_auth:
        return

    if not settings.api_secret_key:
        return

    if not x_api_key or x_api_key != settings.api_secret_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def check_rate_limit(request: Request) -> None:
    """In-memory sliding window rate limiting per client IP."""
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    window_start = current_time - 60.0

    # Purge old records
    recent_requests = [t for t in _request_records[client_ip] if t > window_start]
    _request_records[client_ip] = recent_requests

    if len(recent_requests) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: maximum {settings.rate_limit_per_minute} requests per minute",
        )

    _request_records[client_ip].append(current_time)


def mask_pii(text: Optional[str], show_last_chars: int = 4) -> str:
    """Mask sensitive PII string for secure logs (e.g. DL Number or Phone)."""
    if not text:
        return ""
    if len(text) <= show_last_chars:
        return "*" * len(text)
    return "*" * (len(text) - show_last_chars) + text[-show_last_chars:]

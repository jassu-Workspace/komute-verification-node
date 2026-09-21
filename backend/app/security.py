import secrets
import time
from collections import defaultdict
from typing import Optional
from fastapi import Header, HTTPException, Request, status
from app.config import settings


# In-memory sliding window rate limiter (bounded to prevent memory leaks)
_request_records: dict[str, list[float]] = defaultdict(list)
_MAX_TRACKED_IPS = 2000


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> None:
    """Validate API key header if authentication is enabled with constant-time check."""
    if not settings.enable_api_key_auth:
        return

    if not settings.api_secret_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server authentication misconfigured: API secret key missing in configuration",
        )

    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_secret_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def check_rate_limit(request: Request) -> None:
    """In-memory sliding window rate limiting per client IP with reverse-proxy support and bounded cache."""
    # Extract true client IP behind reverse proxy (Render, Cloudflare, AWS)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    elif request.client and request.client.host:
        client_ip = request.client.host
    else:
        client_ip = "unknown"

    current_time = time.time()
    window_start = current_time - 60.0

    # Purge old records for current IP
    recent_requests = [t for t in _request_records[client_ip] if t > window_start]
    _request_records[client_ip] = recent_requests

    # Bounded cache eviction: prevent memory exhaustion from random IP scans
    if len(_request_records) > _MAX_TRACKED_IPS:
        dead_keys = [k for k, timestamps in _request_records.items() if not timestamps or timestamps[-1] < window_start]
        for k in dead_keys[:500]:
            _request_records.pop(k, None)

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

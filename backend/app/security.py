import secrets
import time
from collections import defaultdict
from typing import Optional
from fastapi import Header, HTTPException, Request, status
from app.config import settings


# In-memory sliding window rate limiter (bounded to prevent memory leaks)
_request_records: dict[str, list[float]] = defaultdict(list)
_MAX_TRACKED_IPS = settings.rate_limit_max_ips


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


import hashlib
import hmac
import re

IDENTIFIER_REGEX = re.compile(r"^[A-Za-z0-9_-]+$")


async def check_rate_limit(request: Request) -> None:
    """In-memory sliding window rate limiting per client IP with reverse-proxy trust validation and bounded cache."""
    direct_ip = request.client.host if (request.client and request.client.host) else "unknown"

    # Only honor X-Forwarded-For if caller connection originates from a trusted reverse proxy
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and direct_ip.lower() in settings.trusted_proxy_set:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = direct_ip

    current_time = time.time()
    window_start = current_time - settings.rate_limit_window_seconds

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
            detail=f"Rate limit exceeded: maximum {settings.rate_limit_per_minute} requests per {settings.rate_limit_window_seconds:.0f} seconds",
        )

    _request_records[client_ip].append(current_time)


def validate_identifier(identifier: str, field_name: str = "identifier") -> str:
    """Validate identifier contains only safe alphanumeric, hyphen, and underscore characters."""
    if not identifier or not IDENTIFIER_REGEX.match(identifier):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {field_name} format. Permitted characters: [A-Za-z0-9_-]",
        )
    return identifier


def verify_driver_ownership(target_driver_id: str, caller_driver_id: Optional[str]) -> None:
    """
    Enforce driver isolation boundary.
    If caller asserts a driver identity via X-Driver-ID header, verify it matches target driver.
    """
    validate_identifier(target_driver_id, "driver_id")
    if caller_driver_id:
        validate_identifier(caller_driver_id, "X-Driver-ID")
        if caller_driver_id != target_driver_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Caller driver '{caller_driver_id}' cannot access records for '{target_driver_id}'.",
            )


def generate_artifact_token(file_path: str, expires_in_seconds: int = 3600) -> str:
    """Generate time-limited HMAC-SHA256 signature token for secure artifact retrieval."""
    secret = (settings.api_secret_key or "kmt_default_secure_secret").encode("utf-8")
    expires_at = int(time.time()) + expires_in_seconds
    msg = f"{file_path}:{expires_at}".encode("utf-8")
    sig = hmac.new(secret, msg, hashlib.sha256).hexdigest()
    return f"{sig}.{expires_at}"


def verify_artifact_token(file_path: str, token: Optional[str]) -> bool:
    """Verify time-limited HMAC-SHA256 signature token."""
    if not token or not isinstance(token, str) or "." not in token:
        return False
    try:
        sig, expires_at_str = token.split(".", 1)
        expires_at = int(expires_at_str)
        if time.time() > expires_at:
            return False
        secret = (settings.api_secret_key or "kmt_default_secure_secret").encode("utf-8")
        msg = f"{file_path}:{expires_at}".encode("utf-8")
        expected_sig = hmac.new(secret, msg, hashlib.sha256).hexdigest()
        return secrets.compare_digest(sig, expected_sig)
    except Exception:
        return False



def mask_pii(text: Optional[str], show_last_chars: int = 4) -> str:
    """Mask sensitive PII string for secure logs (e.g. DL Number or Phone)."""
    if not text:
        return ""
    if len(text) <= show_last_chars:
        return "*" * len(text)
    return "*" * (len(text) - show_last_chars) + text[-show_last_chars:]

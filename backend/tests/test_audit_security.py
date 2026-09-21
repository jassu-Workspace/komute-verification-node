import pytest
from fastapi import HTTPException
from app.config import settings
from app.security import verify_api_key, check_rate_limit, _request_records
from storage.storage import sanitize_folder_name
from core.image_utils import decode_base64_to_image


@pytest.mark.asyncio
async def test_auth_missing_key_when_enabled(monkeypatch):
    """Verify that enabling auth without an API secret key raises 500 error."""
    monkeypatch.setattr(settings, "enable_api_key_auth", True)
    monkeypatch.setattr(settings, "api_secret_key", "")
    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key("some_key")
    assert exc_info.value.status_code == 500
    assert "Server authentication misconfigured" in exc_info.value.detail


@pytest.mark.asyncio
async def test_auth_invalid_key(monkeypatch):
    """Verify constant-time comparison rejects invalid key with 401."""
    monkeypatch.setattr(settings, "enable_api_key_auth", True)
    monkeypatch.setattr(settings, "api_secret_key", "super_secret_token_123")
    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key("wrong_token")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_auth_valid_key(monkeypatch):
    """Verify valid key passes through."""
    monkeypatch.setattr(settings, "enable_api_key_auth", True)
    monkeypatch.setattr(settings, "api_secret_key", "super_secret_token_123")
    # Should not raise
    await verify_api_key("super_secret_token_123")


def test_sanitize_folder_traversal_attacks():
    """Verify all directory traversal vectors are completely neutralized."""
    vectors = [
        ("../../etc/passwd", "etc_passwd"),
        ("..\\..\\windows\\system32", "windows_system32"),
        ("/var/log/syslog", "var_log_syslog"),
        ("....//....//test", "test"),
        ("..", "unknown"),
        ("...", "unknown"),
        ("././.", "unknown"),
        ("valid_driver_123", "valid_driver_123"),
    ]
    for raw, expected in vectors:
        cleaned = sanitize_folder_name(raw)
        assert ".." not in cleaned
        assert "/" not in cleaned
        assert "\\" not in cleaned
        assert cleaned == expected


def test_decode_base64_max_size_enforcement():
    """Verify oversized base64 payload is immediately rejected."""
    oversized = "A" * (31 * 1024 * 1024)
    with pytest.raises(ValueError) as exc:
        decode_base64_to_image(oversized)
    assert "exceeds maximum permitted size" in str(exc.value)


@pytest.mark.asyncio
async def test_rate_limiter_x_forwarded_for():
    """Verify rate limiter extracts true client IP behind reverse proxy."""
    class DummyClient:
        host = "10.0.0.1"

    class DummyRequest:
        headers = {"x-forwarded-for": "203.0.113.45, 10.0.0.1"}
        client = DummyClient()

    # Clear records for this test IP
    _request_records.clear()
    req = DummyRequest()
    await check_rate_limit(req)
    assert "203.0.113.45" in _request_records

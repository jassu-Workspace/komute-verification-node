import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings


@pytest.fixture
def client():
    headers = {}
    if settings.enable_api_key_auth and settings.api_secret_key:
        headers["X-API-Key"] = settings.api_secret_key
    return TestClient(app, headers=headers)


"""
Phase 0/1/6 API-level tests: degraded-subsystem reporting, health readiness,
VLM diagnostics, and the security fixes that are not covered by the VLM unit
tests (path traversal, timeout budget invariant, env precedence).
"""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture
def client():
    headers = {}
    if settings.enable_api_key_auth and settings.api_secret_key:
        headers["X-API-Key"] = settings.api_secret_key
    return TestClient(app, headers=headers)


def _payload():
    import cv2
    import numpy as np

    from core.image_utils import encode_image_to_base64

    dl = np.full((380, 600, 3), 220, dtype=np.uint8)
    cv2.circle(dl, (110, 170), 45, (180, 200, 230), -1)
    cv2.circle(dl, (95, 160), 6, (50, 50, 50), -1)
    cv2.circle(dl, (125, 160), 6, (50, 50, 50), -1)
    cv2.putText(dl, "DL NO: DDNPV1331Q", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(dl, "NAME: Test Driver", (200, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(dl, "DOB: 2005-12-14", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(dl, "EXP: 2030-12-14", (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    selfie = np.full((400, 400, 3), 40, dtype=np.uint8)
    cv2.circle(selfie, (200, 200), 80, (180, 200, 230), -1)
    cv2.circle(selfie, (170, 180), 8, (50, 50, 50), -1)
    cv2.circle(selfie, (230, 180), 8, (50, 50, 50), -1)

    veh = np.full((400, 600, 3), 20, dtype=np.uint8)
    cv2.rectangle(veh, (200, 240), (400, 300), (255, 255, 255), -1)
    cv2.putText(veh, "DL7CQ1939", (210, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return {
        "request_id": "req_diag_001",
        "driver_id": "drv_diag_001",
        "personal_info": {
            "full_name": "Test Driver",
            "date_of_birth": "2005-12-14",
            "gender": "Male",
        },
        "license_details": {
            "license_number": "DDNPV1331Q",
            "issuing_province": "ON",
            "license_expiry_date": "2030-12-14",
        },
        "vehicle_details": {
            "make": "Hyundai",
            "model": "Venue",
            "year": "2020",
            "color": "Black",
            "plate": "DL 7CQ 1939",
        },
        "images": {
            "selfie_base64": encode_image_to_base64(selfie),
            "license_image_base64": encode_image_to_base64(dl),
            "vehicle_photo_base64": encode_image_to_base64(veh),
        },
    }


# ---------------------------------------------------------------------------
# Phase 1: degraded subsystem reporting
# ---------------------------------------------------------------------------


def test_verify_reports_service_status_ok_when_vlm_works(
    client, mock_vlm_provider, detected_face_crops
):
    response = client.post("/api/v1/verify", json=_payload())
    assert response.status_code == 200
    data = response.json()

    assert "service_status" in data
    assert "degraded_subsystems" in data
    assert data["service_status"] == "ok"
    assert data["degraded_subsystems"] == []
    assert data["stages"]["face_biometrics"]["verdict"] != "SERVICE_UNAVAILABLE"
    assert data["stages"]["face_biometrics"]["vlm_provider_used"] == "mock"


def test_verify_reports_degraded_when_vlm_unavailable(
    client, mock_vlm_provider, detected_face_crops
):
    """HTTP stays 200, but service_status makes the outage unmistakable."""
    mock_vlm_provider.configure(configured=False)

    response = client.post("/api/v1/verify", json=_payload())
    assert response.status_code == 200
    data = response.json()

    assert data["decision"] == "REJECTED"
    assert data["service_status"] == "degraded"
    assert len(data["degraded_subsystems"]) == 1

    entry = data["degraded_subsystems"][0]
    assert entry["name"] == "cloud_vlm_biometrics"
    assert entry["status"] == "degraded"
    assert entry["reason"] == "no_credentials"
    assert entry["last_error_at"] is not None

    assert data["stages"]["face_biometrics"]["verdict"] == "SERVICE_UNAVAILABLE"
    assert data["stages"]["face_biometrics"]["passed"] is False
    # Fail-closed: confidence must be zero, never a guessed value.
    assert data["stages"]["face_biometrics"]["vlm_confidence"] == 0.0
    assert data["composite_confidence"] < settings.composite_approval_threshold


def test_degraded_reason_classification_surfaces_upstream_errors(
    client, mock_vlm_provider, detected_face_crops
):
    mock_vlm_provider.configure(
        raw_text="I am unable to compare these images."
    )
    mock_vlm_provider.state["configured"] = True

    response = client.post("/api/v1/verify", json=_payload())
    data = response.json()

    assert data["service_status"] == "degraded"
    entry = data["degraded_subsystems"][0]
    assert entry["reason"] == "unparseable_response"
    assert data["stages"]["face_biometrics"]["verdict"] == "SERVICE_UNAVAILABLE"


def test_response_never_leaks_a_credential(
    client, mock_vlm_provider, detected_face_crops
):
    mock_vlm_provider.configure(configured=False)
    body = client.post("/api/v1/verify", json=_payload()).text

    for secret in (settings.anthropic_api_key,
                   settings.vlm_api_key, settings.api_secret_key):
        if secret and len(secret) > 8:
            assert secret not in body, "a raw credential leaked into the API response"


# ---------------------------------------------------------------------------
# Phase 0: diagnostics + health
# ---------------------------------------------------------------------------


def test_health_reports_vlm_fields(client):
    data = client.get("/health").json()
    for field in (
        "vlm_provider",
        "vlm_model",
        "vlm_configured",
        "vlm_ready",
        "vlm_key_shape_valid",
        "vlm_last_error",
    ):
        assert field in data, f"missing health field {field}"
    assert data["status"] in ("healthy", "degraded")


def test_liveness_excludes_vlm_from_status(client):
    """A VLM outage must not cause the orchestrator to restart the container."""
    liveness = client.get("/live").json()
    assert liveness["status"] in ("healthy", "degraded")
    # Liveness reports VLM state for visibility but does not gate on it.
    assert "vlm_ready" in liveness


def test_diagnostics_endpoint_shape(client):
    response = client.get("/api/v1/diagnostics/vlm")
    assert response.status_code == 200
    data = response.json()

    assert data["primary_provider"] == settings.vlm_provider
    assert isinstance(data["provider_fallback_chain"], list)
    assert data["provider_fallback_chain"][0] == settings.vlm_provider
    assert isinstance(data["providers"], list)
    assert "env_sources" in data
    assert "timeout_budget" in data
    assert "live" not in data or data["live"] is None

    for provider in data["providers"]:
        assert "key_shape_valid" in provider
        assert "key_shape_hint" in provider
        # Masked only: never a raw key.
        masked = provider.get("key_masked")
        if masked and "..." in masked:
            assert settings.vlm_api_key not in masked


def test_diagnostics_reports_env_precedence(client):
    data = client.get("/api/v1/diagnostics/vlm").json()
    env_sources = data["env_sources"]
    assert "loaded_files" in env_sources
    assert "precedence_order" in env_sources
    # A given file must never be reported twice (cwd and backend can collide).
    assert len(env_sources["loaded_files"]) == len(set(env_sources["loaded_files"]))


def test_key_shape_hint_is_suppressed_once_a_provider_is_verified():
    """
    A gateway key of an unexpected shape works fine. Real traffic is ground
    truth, so the advisory shape warning must be dropped once a provider has
    actually returned a valid verdict - otherwise operators rotate a good key.
    """
    from core.vlm_diagnostics import probe_provider, vlm_diagnostics

    if not settings.vlm_api_key:
        pytest.skip("no gateway credential configured")

    entry = probe_provider("openai_compatible")
    original = set(vlm_diagnostics.verified_providers)
    try:
        vlm_diagnostics.verified_providers.clear()
        before = vlm_diagnostics.probe_all()["key_shape_mismatches"]
        if not entry["key_shape_valid"]:
            assert "openai_compatible" in before

            vlm_diagnostics.record_success("openai_compatible")
            after = vlm_diagnostics.probe_all()["key_shape_mismatches"]
            assert "openai_compatible" not in after
            assert "openai_compatible" in vlm_diagnostics.probe_all()["verified_providers"]
    finally:
        vlm_diagnostics.verified_providers.clear()
        vlm_diagnostics.verified_providers.update(original)


# ---------------------------------------------------------------------------
# Phase 4: timeout budget invariant
# ---------------------------------------------------------------------------


def test_vlm_budget_fits_inside_pipeline_budget():
    from app.config import VLM_BUDGET_CEILING

    assert settings.vlm_total_timeout_seconds <= (
        settings.pipeline_timeout_seconds * VLM_BUDGET_CEILING
    )


def test_vlm_per_call_timeout_fits_inside_vlm_budget():
    assert settings.vlm_call_timeout_seconds <= settings.vlm_total_timeout_seconds


def test_enforce_timeout_budget_clamps_when_misconfigured(monkeypatch):
    from app.config import enforce_timeout_budget

    original = settings.vlm_total_timeout_seconds
    try:
        settings.vlm_total_timeout_seconds = 999.0
        report = enforce_timeout_budget()
        assert report["clamped"] is True
        assert settings.vlm_total_timeout_seconds <= (
            settings.pipeline_timeout_seconds * 0.6
        )
    finally:
        settings.vlm_total_timeout_seconds = original


# ---------------------------------------------------------------------------
# Phase 7: security
# ---------------------------------------------------------------------------


def test_storage_route_rejects_path_traversal(client):
    for attempt in (
        "../../../etc/passwd",
        "..%2f..%2f..%2fetc%2fpasswd",
        "..\\..\\..\\windows\\win.ini",
    ):
        response = client.get(f"/api/v1/storage/{attempt}")
        assert response.status_code in (403, 404), (
            f"traversal attempt returned {response.status_code}: {attempt}"
        )
        assert "root:" not in response.text


def test_storage_route_rejects_sibling_directory_escape():
    """
    REGRESSION LOCK: the historical guard was
    `str(target).startswith(str(base_dir))`, which accepts a sibling directory
    that shares the string prefix, e.g.
    base=".../storage/uploads" vs target=".../storage/uploads_evil/secret.txt".

    Invoked directly on the handler because httpx normalises "../" segments
    before the request ever reaches the router.
    """
    import asyncio

    from fastapi import HTTPException

    from app.config import settings as s
    from app.routes import get_storage_file

    sibling = s.uploads_root.parent / (s.uploads_root.name + "_evil")
    sibling.mkdir(parents=True, exist_ok=True)
    secret = sibling / "secret.txt"
    secret.write_text("TOP-SECRET-CANARY")

    try:
        escape_paths = [
            f"../{s.uploads_root.name}_evil/secret.txt",
            f"..\\{s.uploads_root.name}_evil\\secret.txt",
            "../../etc/passwd",
        ]
        for path in escape_paths:
            with pytest.raises(HTTPException) as excinfo:
                asyncio.run(get_storage_file(path))
            assert excinfo.value.status_code == 403, (
                f"{path!r} was not rejected with 403 "
                f"(got {excinfo.value.status_code})"
            )
            assert "TOP-SECRET-CANARY" not in str(excinfo.value.detail)
    finally:
        secret.unlink(missing_ok=True)
        try:
            sibling.rmdir()
        except OSError:
            pass


def test_storage_route_serves_legitimate_files(client):
    """The traversal guard must not break normal artifact serving."""
    root = settings.uploads_root
    probe_dir = root / "previews" / "pytest_canary"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_dir / "note.txt"
    probe.write_text("ok")
    try:
        response = client.get("/api/v1/storage/previews/pytest_canary/note.txt")
        assert response.status_code == 200
        assert "ok" in response.text
    finally:
        probe.unlink(missing_ok=True)
        try:
            probe_dir.rmdir()
        except OSError:
            pass


def test_cors_is_not_wildcard_with_credentials():
    """allow_origins=['*'] + allow_credentials=True is an unsafe pairing."""
    from fastapi.middleware.cors import CORSMiddleware

    registered = False
    for mw in app.user_middleware:
        if mw.cls is CORSMiddleware:
            registered = True
            assert mw.kwargs.get("allow_origins") != ["*"]
            assert not (
                mw.kwargs.get("allow_credentials") and mw.kwargs.get("allow_origins") == ["*"]
            )
    assert registered, "CORS middleware not registered"


def test_cors_actually_blocks_a_foreign_origin_by_default():
    """
    Behavioural check, not just a config inspection: with CORS_ALLOW_ORIGINS
    unset, a browser-style cross-origin request must not be granted access.
    """
    from fastapi.testclient import TestClient

    if settings.resolved_cors_origins:
        pytest.skip("CORS origins are configured, so the restrictive default is not active")

    local = TestClient(app)
    response = local.get("/ping", headers={"Origin": "https://evil.example.com"})
    assert response.status_code == 200
    # No Access-Control-Allow-Origin header means the browser blocks the read.
    assert "access-control-allow-origin" not in {
        k.lower() for k in response.headers
    }


def test_cors_allows_a_configured_origin(monkeypatch):
    """When an allowlist is configured, that origin is granted and others are not."""
    from fastapi.middleware.cors import CORSMiddleware

    monkeypatch.setattr(settings, "cors_allow_origins", "https://app.komute.com")

    fresh = FastAPI()
    fresh.add_middleware(
        CORSMiddleware,
        allow_origins=settings.resolved_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fresh.get("/ping")
    async def _ping():
        return {"status": "ok"}

    client = TestClient(fresh)

    allowed = client.get("/ping", headers={"Origin": "https://app.komute.com"})
    assert allowed.headers.get("access-control-allow-origin") == "https://app.komute.com"

    blocked = client.get("/ping", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in {
        k.lower() for k in blocked.headers
    }


# ---------------------------------------------------------------------------
# Phase 5: secrets must never be baked into the image
# ---------------------------------------------------------------------------


def _dockerignore_patterns():
    from app.config import REPO_ROOT

    path = REPO_ROOT / ".dockerignore"
    if not path.is_file():
        pytest.skip(".dockerignore not found")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_dockerignore_blocks_env_and_credential_files():
    """
    `COPY . .` in the Dockerfile means anything not excluded lands in an image
    layer, where it survives layer history even if deleted later.
    """
    import re

    patterns = _dockerignore_patterns()

    def compile_patterns():
        compiled = []
        for raw in patterns:
            negate = raw.startswith("!")
            pat = raw.lstrip("/")[1:] if negate else raw.lstrip("/")
            dir_only = pat.endswith("/")
            if dir_only:
                pat = pat[:-1]
            out, i = "", 0
            while i < len(pat):
                if pat.startswith("**/", i):
                    out += "(?:.*/)?"
                    i += 3
                elif pat.startswith("**", i):
                    out += ".*"
                    i += 2
                elif pat[i] == "*":
                    out += "[^/]*"
                    i += 1
                elif pat[i] == "?":
                    out += "[^/]"
                    i += 1
                else:
                    out += re.escape(pat[i])
                    i += 1
            out = "(?:^|.*/)" + out
            out = out + ("(?:/.*)?$" if dir_only else "(?:/.*)?$")
            compiled.append((re.compile(out), negate))
        return compiled

    compiled = compile_patterns()

    def excluded(path: str) -> bool:
        state = False
        for regex, negate in compiled:
            if regex.match(path):
                state = not negate
        return state

    for secret_path in (
        ".env",
        "backend/.env",
        ".env.local",
        "backend/.env.production",
        "id_rsa.pem",
        "server.key",
        "azure.p12",
        "secrets/db.json",
    ):
        assert excluded(secret_path), f"{secret_path} would be baked into the image"

    # The template must survive, or there is nothing to copy from.
    assert not excluded(".env.example"), ".env.example must stay in the image"

    # And the application itself must not be excluded.
    for source_path in (
        "backend/app/config.py",
        "backend/app/main.py",
        "backend/core/vlm_providers.py",
        "backend/core/vlm_diagnostics.py",
        "requirements.txt",
        "Dockerfile",
    ):
        assert not excluded(source_path), f"{source_path} must be in the image"

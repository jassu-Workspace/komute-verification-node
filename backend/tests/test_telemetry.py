import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_system_page_endpoint():
    """Verify /system and /telemetry HTML rendering."""
    for path in ["/system", "/telemetry"]:
        response = client.get(path)
        assert response.status_code == 200
        assert "Komüte Telemetry Hub" in response.text
        assert "AI & CV Engine Matrix" in response.text


def test_telemetry_system_api():
    """Verify /api/v1/telemetry/system returns structured segregated JSON."""
    response = client.get("/api/v1/telemetry/system")
    assert response.status_code == 200
    data = response.json()

    assert "vitals" in data
    assert "libraries" in data
    assert "ai_models" in data
    assert "pipeline_config" in data

    # Check vitals structure
    assert "application" in data["vitals"]
    assert "runtime" in data["vitals"]
    assert "storage" in data["vitals"]
    assert data["vitals"]["runtime"]["python_version"] is not None

    # Check AI models list (should contain 4 optimized engines)
    models = data["ai_models"]
    assert len(models) >= 4
    model_ids = [m["id"] for m in models]
    assert "rapidocr_onnx" in model_ids
    assert "yunet_dnn" in model_ids
    assert "cloud_vlm" in model_ids


def test_telemetry_sessions_api():
    """Verify /api/v1/telemetry/sessions returns past session lists."""
    response = client.get("/api/v1/telemetry/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "sessions" in data
    assert isinstance(data["sessions"], list)


def test_telemetry_self_test_api():
    """Verify /api/v1/telemetry/self-test executes live benchmark benchmarks."""
    response = client.post("/api/v1/telemetry/self-test")
    assert response.status_code == 200
    data = response.json()
    assert "overall_status" in data
    assert "components" in data
    assert "face_privacy_cropper" in data["components"]
    assert "image_compression_engine" in data["components"]

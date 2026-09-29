"""
Phase 2/6: provider transport tests.

The OpenAI-compatible and Anthropic providers could not be exercised against a
real endpoint in CI, so their request shape and response handling are pinned
here with an httpx MockTransport and a stubbed Anthropic client. Without these,
the entire provider-failover story (the reason the service survives a throttled
model) is untested.
"""

import json

import httpx
import pytest

from app.config import settings
from core.vlm_providers import (
    AnthropicProvider,
    BiometricVerdictValidationError,
    OpenAICompatibleProvider,
    close_shared_httpx_client,
    get_shared_httpx_client,
)


def _verdict_payload(match=True, confidence=0.93, live=True, verdict="MATCH_CONFIRMED"):
    return {
        "is_match": match,
        "confidence_score": confidence,
        "is_live_selfie": live,
        "estimated_age_delta_years": 2,
        "facial_feature_notes": "Identical nasal bridge.",
        "reasoning": "Same individual.",
        "verdict": verdict,
    }


@pytest.fixture(autouse=True)
async def _noop_shared_client():
    yield
    await close_shared_httpx_client()


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------


def test_openai_compat_requires_all_three_settings():
    original = (
        settings.vlm_api_base_url,
        settings.vlm_api_key,
        settings.vlm_openai_model,
    )
    try:
        settings.vlm_api_base_url = None
        settings.vlm_api_key = "sk-test"
        settings.vlm_openai_model = "qwen-vl"
        assert OpenAICompatibleProvider().is_configured() is False

        settings.vlm_api_base_url = "http://localhost:8000/v1"
        settings.vlm_api_key = None
        assert OpenAICompatibleProvider().is_configured() is False

        settings.vlm_api_key = "sk-test"
        settings.vlm_openai_model = None
        assert OpenAICompatibleProvider().is_configured() is False

        settings.vlm_openai_model = "qwen-vl"
        assert OpenAICompatibleProvider().is_configured() is True
    finally:
        (
            settings.vlm_api_base_url,
            settings.vlm_api_key,
            settings.vlm_openai_model,
        ) = original


@pytest.mark.asyncio
async def test_openai_compat_sends_correct_request_shape():
    """Two image_url parts + text, bearer auth, json_object response_format."""
    original = (
        settings.vlm_api_base_url,
        settings.vlm_api_key,
        settings.vlm_openai_model,
    )
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(_verdict_payload())}}]}
        )

    try:
        settings.vlm_api_base_url = "http://vllm.internal:8000/v1/"
        settings.vlm_api_key = "sk-test-key"
        settings.vlm_openai_model = "Qwen2.5-VL-7B-Instruct"

        import core.vlm_providers as vp

        transport_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        original_get = vp.get_shared_httpx_client
        vp.get_shared_httpx_client = lambda: transport_client
        try:
            provider = OpenAICompatibleProvider()
            assert provider.is_configured()
            result = await provider.verify("QUFB", "QkJC")
        finally:
            vp.get_shared_httpx_client = original_get
            await transport_client.aclose()
    finally:
        (
            settings.vlm_api_base_url,
            settings.vlm_api_key,
            settings.vlm_openai_model,
        ) = original

    # Trailing slash on the base URL must not double up.
    assert captured["url"] == "http://vllm.internal:8000/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test-key"

    body = captured["body"]
    assert body["model"] == "Qwen2.5-VL-7B-Instruct"
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["role"] == "system"
    assert "MATCH_CONFIRMED" in body["messages"][0]["content"]

    parts = body["messages"][1]["content"]
    assert len(parts) == 3
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"] == "data:image/jpeg;base64,QUFB"
    assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,QkJC"
    assert parts[2]["type"] == "text"

    # And the response was actually validated, not just passed through.
    assert result.verdict is not None
    assert result.verdict.is_match is True
    assert result.verdict.confidence_score == 0.93
    assert result.provider == "openai_compatible"


@pytest.mark.asyncio
async def test_openai_compat_unwraps_sse_framed_body():
    """Gateways may return text/event-stream with `data:` lines for non-streaming calls."""
    original = (
        settings.vlm_api_base_url,
        settings.vlm_api_key,
        settings.vlm_openai_model,
    )
    verdict_json = json.dumps(_verdict_payload())
    envelope = json.dumps({"choices": [{"message": {"content": verdict_json}}]})
    sse_body = (
        "\n\n"
        f"data: {envelope}\n\n"
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=sse_body,
        )

    try:
        settings.vlm_api_base_url = "https://our-llm.onrender.com/v1"
        settings.vlm_api_key = "sk-test-key"
        settings.vlm_openai_model = "openrouter/stealth/space-bunny-alpha"

        import core.vlm_providers as vp

        transport_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        original_get = vp.get_shared_httpx_client
        vp.get_shared_httpx_client = lambda: transport_client
        try:
            provider = OpenAICompatibleProvider()
            result = await provider.verify("QUFB", "QkJC")
        finally:
            vp.get_shared_httpx_client = original_get
            await transport_client.aclose()
    finally:
        (
            settings.vlm_api_base_url,
            settings.vlm_api_key,
            settings.vlm_openai_model,
        ) = original

    assert result.verdict is not None
    assert result.verdict.is_match is True
    assert result.provider == "openai_compatible"


def test_parse_chat_body_rejects_empty_and_malformed():
    from core.vlm_providers import _parse_chat_body

    with pytest.raises(RuntimeError):
        _parse_chat_body("")
    with pytest.raises(RuntimeError):
        _parse_chat_body("data: [DONE]\n\n")
    with pytest.raises(RuntimeError):
        _parse_chat_body("{not json")
    with pytest.raises(RuntimeError):
        _parse_chat_body("[1, 2]")  # non-object body

    # Trailing junk after a complete object (`{...}data: [DONE]`).
    obj = {"choices": [{"message": {"content": "hi"}}]}
    assert _parse_chat_body(json.dumps(obj) + "data: [DONE]") == obj


def test_parse_chat_body_merges_streaming_deltas():
    """claude-all-mix streams chat.completion.chunk deltas with no message object."""
    from core.vlm_providers import _parse_chat_body

    chunk1 = json.dumps({
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    })
    chunk2 = json.dumps({
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": '{"verdict": '}, "finish_reason": None}],
    })
    chunk3 = json.dumps({
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": '"MISMATCH"}'}, "finish_reason": "stop"}],
    })
    body = _parse_chat_body(f"data: {chunk1}\n\ndata: {chunk2}\n\ndata: {chunk3}\n\ndata: [DONE]\n\n")
    assert body["choices"][0]["message"]["content"] == '{"verdict": "MISMATCH"}'


def test_extract_message_text_shapes():
    from core.vlm_providers import _extract_message_text

    assert _extract_message_text({"choices": [{"message": {"content": "hi"}}]}) == "hi"
    assert _extract_message_text(
        {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}
    ) == "ab"
    assert _extract_message_text({"choices": [{"text": "legacy"}]}) == "legacy"
    with pytest.raises(RuntimeError):
        _extract_message_text({"choices": [{"message": {"content": ""}}]})
    with pytest.raises(RuntimeError):
        _extract_message_text({"choices": []})


@pytest.mark.asyncio
async def test_openai_compat_http_error_surfaces_status():
    original = (
        settings.vlm_api_base_url,
        settings.vlm_api_key,
        settings.vlm_openai_model,
    )
    try:
        settings.vlm_api_base_url = "http://vllm.internal:8000/v1"
        settings.vlm_api_key = "sk-test-key"
        settings.vlm_openai_model = "qwen-vl"

        import core.vlm_providers as vp

        transport_client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(401, text="nope"))
        )
        original_get = vp.get_shared_httpx_client
        vp.get_shared_httpx_client = lambda: transport_client
        try:
            with pytest.raises(Exception) as excinfo:
                await OpenAICompatibleProvider().verify("QUFB", "QkJC")
        finally:
            vp.get_shared_httpx_client = original_get
            await transport_client.aclose()
    finally:
        (
            settings.vlm_api_base_url,
            settings.vlm_api_key,
            settings.vlm_openai_model,
        ) = original

    message = str(excinfo.value)
    assert "401" in message

    from core.vlm_diagnostics import classify_error

    assert classify_error(excinfo.value) == "invalid_credentials"


@pytest.mark.asyncio
async def test_openai_compat_connectivity_check_uses_models_endpoint():
    original = (
        settings.vlm_api_base_url,
        settings.vlm_api_key,
        settings.vlm_openai_model,
    )
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"data": []})

    try:
        settings.vlm_api_base_url = "http://vllm.internal:8000/v1"
        settings.vlm_api_key = "sk-test-key"
        settings.vlm_openai_model = "qwen-vl"

        import core.vlm_providers as vp

        transport_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        original_get = vp.get_shared_httpx_client
        vp.get_shared_httpx_client = lambda: transport_client
        try:
            await OpenAICompatibleProvider().connectivity_check()
        finally:
            vp.get_shared_httpx_client = original_get
            await transport_client.aclose()
    finally:
        (
            settings.vlm_api_base_url,
            settings.vlm_api_key,
            settings.vlm_openai_model,
        ) = original

    assert seen["url"] == "http://vllm.internal:8000/v1/models"


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


class _StubBlock:
    def __init__(self, text):
        self.text = text


class _StubResponse:
    def __init__(self, text):
        self.content = [_StubBlock(text)]


class _StubMessages:
    def __init__(self, sink):
        self.sink = sink

    async def create(self, **kwargs):
        self.sink.append(kwargs)
        return _StubResponse(json.dumps(_verdict_payload(confidence=0.9)))


class _StubAnthropicClient:
    def __init__(self, api_key=None):
        self.sink = []
        self.messages = _StubMessages(self.sink)


@pytest.mark.asyncio
async def test_anthropic_sends_correct_request_shape_and_parses():
    original_key = settings.anthropic_api_key
    try:
        settings.anthropic_api_key = "sk-ant-test"
        provider = AnthropicProvider()
        provider._client = _StubAnthropicClient()

        assert provider.is_configured() is True
        assert provider.model == "claude-sonnet-5"

        result = await provider.verify("QUFB", "QkJC")
    finally:
        settings.anthropic_api_key = original_key

    sent = provider._client.sink[0]
    # The retired claude-3-5-sonnet must never be sent.
    assert sent["model"] == "claude-sonnet-5"
    assert "claude-3-5-sonnet" not in sent["model"]

    # The strict system prompt plus the repair instruction must be present.
    assert "MATCH_CONFIRMED" in sent["system"]
    assert "no markdown fences" in sent["system"].lower()

    blocks = sent["messages"][0]["content"]
    assert len(blocks) == 3
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["media_type"] == "image/jpeg"
    assert blocks[0]["source"]["data"] == "QUFB"
    assert blocks[1]["source"]["data"] == "QkJC"
    assert blocks[2]["type"] == "text"

    assert result.verdict.is_match is True
    assert result.verdict.confidence_score == 0.9
    assert result.provider == "anthropic"


def test_anthropic_unconfigured_when_key_absent():
    original = settings.anthropic_api_key
    try:
        settings.anthropic_api_key = None
        assert AnthropicProvider().is_configured() is False
    finally:
        settings.anthropic_api_key = original


# ---------------------------------------------------------------------------
# Retry-on-unparseable (Phase 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unparseable_response_is_retried_then_succeeds():
    """
    vlm_parse_retry_count must actually re-ask. A prose reply followed by a
    valid JSON reply should succeed, not fail closed.
    """
    calls = []

    from core.vlm_providers import VLMProvider

    class _Flaky(VLMProvider):
        name = "mock"

        @property
        def model(self):
            return "mock"

        def is_configured(self):
            return True

        async def connectivity_check(self):
            return None

        async def _call_once(self, selfie_b64, dl_b64, repair_text=""):
            calls.append(repair_text)
            if len(calls) == 1:
                return "I cannot determine if the images match."
            return json.dumps(_verdict_payload())

    original_count = settings.vlm_parse_retry_count
    try:
        settings.vlm_parse_retry_count = 1
        result = await _Flaky().verify("QUFB", "QkJC")
    finally:
        settings.vlm_parse_retry_count = original_count

    assert len(calls) == 2
    # The retry must carry the offending text back to the model.
    assert "cannot determine" in calls[1]
    assert result.verdict.is_match is True
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_unparseable_exhausts_retries_then_fails_closed():
    calls = []

    from core.vlm_providers import VLMProvider

    class _AlwaysProse(VLMProvider):
        name = "mock"

        @property
        def model(self):
            return "mock"

        def is_configured(self):
            return True

        async def connectivity_check(self):
            return None

        async def _call_once(self, selfie_b64, dl_b64, repair_text=""):
            calls.append(repair_text)
            return "These two people are clearly different people."

    original_count = settings.vlm_parse_retry_count
    try:
        settings.vlm_parse_retry_count = 1
        with pytest.raises(BiometricVerdictValidationError):
            await _AlwaysProse().verify("QUFB", "QkJC")
    finally:
        settings.vlm_parse_retry_count = original_count

    assert len(calls) == 2  # initial + 1 retry


# ---------------------------------------------------------------------------
# Model-level fallback clones
# ---------------------------------------------------------------------------


def test_openai_compat_with_model_clones_share_the_client():
    """with_model() must not rebuild the HTTP client for every fallback."""
    from core.vlm_providers import OpenAICompatibleProvider

    original = (settings.vlm_api_key, settings.vlm_api_base_url, settings.vlm_openai_model)
    try:
        settings.vlm_api_key = "sk-stub-for-tests-only"
        settings.vlm_api_base_url = "https://our-llm.onrender.com/v1"
        settings.vlm_openai_model = "primary-model"
        provider = OpenAICompatibleProvider()
        clone = provider.with_model("secondary-model")
    finally:
        settings.vlm_api_key, settings.vlm_api_base_url, settings.vlm_openai_model = original

    assert clone.model == "secondary-model"
    assert clone._base == provider._base
    assert clone._key == provider._key
    assert clone is not provider


def test_unknown_provider_is_rejected():
    from core.vlm_providers import build_provider

    with pytest.raises(ValueError):
        build_provider("gemini")


# ---------------------------------------------------------------------------
# Shared HTTP client lifecycle (Phase 4)
# ---------------------------------------------------------------------------


def test_shared_httpx_client_is_reused():
    a = get_shared_httpx_client()
    b = get_shared_httpx_client()
    assert a is b, "a new client per request would leak connection pools"


@pytest.mark.asyncio
async def test_shared_httpx_client_closes_and_resets():
    client = get_shared_httpx_client()
    await close_shared_httpx_client()
    assert client.is_closed
    assert get_shared_httpx_client() is not client
    await close_shared_httpx_client()


def test_lifespan_shutdown_runs_and_closes_the_shared_client():
    """
    The lifespan shutdown hook must actually execute and release the pooled
    HTTP client. Windows `proc.terminate()` is a hard kill and never exercises
    this, so it is driven through the ASGI lifespan protocol directly.
    """
    import anyio
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    client_holder = {}

    async def _drive():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as http:
            async with app.router.lifespan_context(app):
                # Materialise the shared pool while the app is "running".
                client_holder["pool"] = get_shared_httpx_client()
                assert not client_holder["pool"].is_closed
                probe = await http.get("/live")
                assert probe.status_code == 200
        return True

    anyio.run(_drive)

    assert client_holder["pool"].is_closed, "lifespan shutdown did not close the HTTP pool"
    # And the next process/request rebuilds it rather than reusing a closed one.
    assert get_shared_httpx_client() is not client_holder["pool"]

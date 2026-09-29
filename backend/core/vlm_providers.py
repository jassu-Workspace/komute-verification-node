"""
Phase 2 + 4: unified, fully-async cloud VLM provider layer.

Every provider is natively asynchronous, so no `asyncio.to_thread` is used and
cancellation is real (the historical `wait_for(to_thread(...))` combination
leaked a live thread per model candidate on every request).

Fail-closed by construction: a provider response that does not satisfy
`BiometricVerdict` raises `BiometricVerdictValidationError` and the caller falls
back to a strict rejection. No keyword heuristics, no fail-open defaults.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from app.biometric_schemas import (
    BiometricVerdict,
)
from app.config import settings
from core.vlm_diagnostics import classify_error, vlm_diagnostics

logger = logging.getLogger(__name__)

BIOMETRIC_SYSTEM_PROMPT = """You are a highly strict forensic biometric verification expert. You are provided with two tightly cropped facial images:
- Image 1 is a live user selfie face.
- Image 2 is a government Driving License ID card portrait photo.

Your absolute highest priority task is to determine whether both facial images belong to the EXACT SAME human individual. You must be extremely rigorous and completely intolerant of false positives. If the two images are of different people, you MUST firmly reject the match with a low confidence score (< 0.40).

CRITICAL MATCHING RULES:
1. DIFFERENT PEOPLE: If the facial bone structure, jawline, nose width, or eye spacing are structurally distinct, this is a MISMATCH. Do not give the benefit of the doubt. Reject them immediately.
2. LARGE AGE DIFFERENCES: It is highly common for a driving license to be 10-20 years old. If the core craniofacial bone structure, eye sockets, and nasal bridge geometry match perfectly, but one image has wrinkles, weight gain/loss, grey hair, or age degradation, you MUST still CONFIRM the match. Age deltas do not change bone structure.
3. COSMETIC CHANGES: Ignore changes in facial hair (beards/mustaches), makeup, glasses, lighting, focal distance, or hairstyle. Focus exclusively on underlying skeletal geometry.
4. SPOOFING: If Image 1 is a photo of a screen, a printed photo, a mask, or a deepfake, you MUST set is_live_selfie to false.

Perform deep morphological analysis:
1. Craniofacial bone structure: eye-to-nose-to-mouth triangular proportions and symmetry.
2. Nasal bridge geometry, nostril width, and ear lobe morphology.
3. Inter-pupillary distance ratio and eye socket depth.
4. Liveness and anti-spoof analysis of Image 1.

Respond with a single JSON object and nothing else.
Your JSON response MUST contain exactly these keys:
- "is_match": boolean (true if both images are the exact same individual, false otherwise)
- "confidence_score": number between 0.0 and 1.0 (similarity confidence)
- "is_live_selfie": boolean (true if Image 1 is a genuine live human face, false if spoof/screen/paper)
- "estimated_age_delta_years": integer or null (approximate age difference between ID and selfie)
- "facial_feature_notes": string (brief geometry and landmark notes)
- "reasoning": string (forensic justification)
- "verdict": exactly one of "MATCH_CONFIRMED", "MISMATCH", or "INCONCLUSIVE". Do not use any other label."""

_REPAIR_INSTRUCTION = (
    "Your previous reply was not valid JSON matching the required schema. "
    "Reply again with ONLY a JSON object containing exactly these keys: "
    "is_match (boolean), confidence_score (number 0-1), is_live_selfie (boolean), "
    "estimated_age_delta_years (integer or null), facial_feature_notes (string), "
    "reasoning (string), verdict (one of MATCH_CONFIRMED, MISMATCH, INCONCLUSIVE). "
    "No prose, no markdown fences."
)


class BiometricVerdictValidationError(ValueError):
    """Raised when a provider response cannot be validated. Always fail-closed."""


class ProviderResult:
    """Outcome of a single provider attempt."""

    def __init__(
        self,
        provider: str,
        model: str,
        verdict: Optional[BiometricVerdict],
        latency_ms: int,
        raw_text: str = "",
        attempts: int = 1,
    ) -> None:
        self.provider = provider
        self.model = model
        self.verdict = verdict
        self.latency_ms = latency_ms
        self.raw_text = raw_text
        self.attempts = attempts

    def to_chain_entry(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "ok": self.verdict is not None,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
        }


# --------------------------------------------------------------------------
# Shared HTTP client (Phase 4: one pooled client instead of per-request ones)
# --------------------------------------------------------------------------

_shared_httpx_client: Optional["httpx.AsyncClient"] = None  # type: ignore[name-defined]


def get_shared_httpx_client():
    global _shared_httpx_client
    if _shared_httpx_client is None:
        import httpx

        _shared_httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.vlm_call_timeout_seconds),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _shared_httpx_client


async def close_shared_httpx_client() -> None:
    global _shared_httpx_client
    if _shared_httpx_client is not None:
        await _shared_httpx_client.aclose()
        _shared_httpx_client = None


# --------------------------------------------------------------------------
# Base provider
# --------------------------------------------------------------------------


class VLMProvider:
    name = "base"

    def is_configured(self) -> bool:
        raise NotImplementedError

    async def connectivity_check(self) -> None:
        raise NotImplementedError

    async def _call_once(self, selfie_b64: str, dl_b64: str, repair_text: str = "") -> str:
        raise NotImplementedError

    @property
    def model(self) -> str:
        raise NotImplementedError

    async def verify(self, selfie_b64: str, dl_b64: str) -> ProviderResult:
        """Call the provider, validate the response, retry once on a parse failure."""
        started = time.perf_counter()
        raw = await self._call_once(selfie_b64, dl_b64)
        attempts = 1

        try:
            verdict = self._parse(raw)
        except BiometricVerdictValidationError as first_error:
            for _ in range(max(0, settings.vlm_parse_retry_count)):
                attempts += 1
                logger.warning(
                    "Provider '%s' returned an unparseable biometric response (%s). Retrying with repair instruction.",
                    self.name,
                    first_error,
                )
                raw = await self._call_once(selfie_b64, dl_b64, repair_text=raw)
                try:
                    verdict = self._parse(raw)
                    break
                except BiometricVerdictValidationError as retry_error:
                    first_error = retry_error
            else:
                vlm_diagnostics.record_unparseable(
                    self.name, self.model, str(first_error)
                )
                raise first_error

        latency_ms = int((time.perf_counter() - started) * 1000)
        vlm_diagnostics.record_success(self.name)
        return ProviderResult(
            provider=self.name,
            model=self.model,
            verdict=verdict,
            latency_ms=latency_ms,
            raw_text=raw,
            attempts=attempts,
        )

    def _parse(self, text: str) -> BiometricVerdict:
        """Extract and validate the JSON verdict. Fail-closed on any deviation."""
        if not text or not text.strip():
            raise BiometricVerdictValidationError("empty provider response")

        candidate = text.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
        if fence:
            candidate = fence.group(1)
        else:
            brace = re.search(r"(\{.*\})", candidate, re.DOTALL)
            if brace:
                candidate = brace.group(1)

        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as exc:
            # NOTE: no keyword heuristic here. Historically a prose reply
            # containing the word "match" produced is_match=True / 0.85 /
            # is_live=True and therefore a false APPROVED decision.
            raise BiometricVerdictValidationError(
                f"provider response is not valid JSON: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise BiometricVerdictValidationError(
                f"provider response is {type(payload).__name__}, expected object"
            )

        # Normalise the verdict label; reject anything outside the enum.
        raw_label = str(payload.get("verdict", "")).strip().upper()
        label_aliases = {
            "MATCH": "MATCH_CONFIRMED",
            "MATCH_CONFIRMED": "MATCH_CONFIRMED",
            "MISMATCH": "MISMATCH",
            "NO_MATCH": "MISMATCH",
            "INCONCLUSIVE": "INCONCLUSIVE",
            "UNKNOWN": "INCONCLUSIVE",
        }
        if raw_label in label_aliases:
            payload["verdict"] = label_aliases[raw_label]

        # Extract confidence score from aliases without premature break on missing keys
        confidence = 0.0
        for alias in ("confidence_score", "match_confidence", "confidence", "similarity_score", "score"):
            if alias in payload and payload[alias] is not None:
                val = payload[alias]
                try:
                    if isinstance(val, str) and "%" in val:
                        c = float(val.replace("%", "").strip()) / 100.0
                    else:
                        c = float(val)
                    confidence = c
                    break
                except (TypeError, ValueError):
                    continue
        payload["confidence_score"] = confidence

        # Extract is_match
        if "is_match" in payload:
            raw_match = payload["is_match"]
            if raw_match is None:
                payload["is_match"] = False
            elif isinstance(raw_match, str):
                payload["is_match"] = raw_match.strip().lower() in ("true", "1", "yes")
            else:
                payload["is_match"] = bool(raw_match)
        elif "match" in payload:
            raw_match = payload["match"]
            if raw_match is None:
                payload["is_match"] = False
            elif isinstance(raw_match, str):
                payload["is_match"] = raw_match.strip().lower() in ("true", "1", "yes")
            else:
                payload["is_match"] = bool(raw_match)
        else:
            # Model omitted match key; infer from explicit verdict & confidence
            payload["is_match"] = (
                payload.get("verdict") == "MATCH_CONFIRMED"
                and payload["confidence_score"] >= settings.vlm_min_match_confidence
            )

        # Extract is_live_selfie
        raw_live = None
        for live_alias in ("is_live_selfie", "is_live", "liveness"):
            if live_alias in payload and payload[live_alias] is not None:
                raw_live = payload[live_alias]
                break

        if raw_live is None:
            payload["is_live_selfie"] = False
        elif isinstance(raw_live, str):
            payload["is_live_selfie"] = raw_live.strip().lower() in ("true", "1", "yes", "live")
        else:
            payload["is_live_selfie"] = bool(raw_live)

        for target, aliases in (
            ("reasoning", ("reasoning", "reason", "explanation", "details")),
            ("facial_feature_notes", ("facial_feature_notes", "notes", "features")),
        ):
            if not payload.get(target):
                for alias in aliases:
                    if payload.get(alias):
                        payload[target] = str(payload[alias])[:2000]
                        break
                else:
                    payload[target] = ""

        # Derive verdict if missing or empty
        if not payload.get("verdict"):
            payload["verdict"] = (
                "MATCH_CONFIRMED"
                if payload["is_match"] is True
                and payload["confidence_score"] >= settings.vlm_min_match_confidence
                else "MISMATCH"
            )

        try:
            return BiometricVerdict.model_validate(payload)
        except ValidationError as exc:
            raise BiometricVerdictValidationError(
                f"provider response failed schema validation: {exc.error_count()} error(s)"
            ) from exc


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


class AnthropicProvider(VLMProvider):
    name = "anthropic"

    def __init__(self) -> None:
        self._key = settings.anthropic_api_key
        self._model = settings.anthropic_model
        self._client = None
        if self._key:
            try:
                import anthropic

                self._client = anthropic.AsyncAnthropic(api_key=self._key)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to construct Anthropic client: %s", type(exc).__name__)
                self._client = None

    def is_configured(self) -> bool:
        return bool(self._key) and self._client is not None

    @property
    def model(self) -> str:
        return self._model

    async def connectivity_check(self) -> None:
        if not self.is_configured():
            raise RuntimeError("Anthropic is not configured")
        async with asyncio.timeout(settings.vlm_call_timeout_seconds):
            await self._client.messages.create(
                model=self._model,
                max_tokens=settings.vlm_connectivity_max_tokens,
                messages=[{"role": "user", "content": "ping"}],
            )

    async def _call_once(self, selfie_b64: str, dl_b64: str, repair_text: str = "") -> str:
        if not self.is_configured():
            raise ValueError("ANTHROPIC_API_KEY is not configured")

        text_block = (
            "Image 1 is the live selfie crop, Image 2 is the ID portrait face crop. "
            "Analyse biometric match and liveness."
        )
        if repair_text:
            text_block += f"\n\nYour previous reply was:\n{repair_text[:settings.vlm_repair_context_chars]}"

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=settings.vlm_max_tokens,
            system=BIOMETRIC_SYSTEM_PROMPT + "\n\n" + _REPAIR_INSTRUCTION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": selfie_b64,
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": dl_b64,
                            },
                        },
                        {"type": "text", "text": text_block},
                    ],
                }
            ],
        )
        return "".join(getattr(block, "text", "") for block in response.content)


# --------------------------------------------------------------------------
# OpenAI-compatible (vLLM, OpenRouter, Azure OpenAI, local gateways)
# --------------------------------------------------------------------------


def _parse_chat_body(text: str) -> Dict[str, Any]:
    """Parse a /chat/completions body, tolerating gateway framing quirks.

    Observed shapes from OpenAI-compatible gateways:
      1. Plain JSON object (spec behaviour).
      2. SSE framing: `data: {...}` lines plus `data: [DONE]`, even for
         non-streaming requests (`Content-Type: text/event-stream`).
      3. A JSON object with trailing junk, e.g. `{...}data: [DONE]`.
      4. True streaming chunks (`object: chat.completion.chunk` with
         `choices[].delta.content` fragments and no `message` object).

    A strict `response.json()` fails on shapes 2-4, which would surface
    as a spurious provider outage, so all of them are unwrapped here.
    Streaming deltas are merged into a synthetic complete envelope.
    """
    candidate = (text or "").strip()
    if not candidate:
        raise RuntimeError("openai_compatible returned an empty body")

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Collect every SSE `data:` chunk in order.
    chunks: List[Dict[str, Any]] = []
    for line in candidate.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            chunk_text = line[5:].strip()
            if chunk_text and chunk_text != "[DONE]":
                try:
                    chunk = json.loads(chunk_text)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, dict):
                    chunks.append(chunk)

    # A complete (non-chunk) envelope wins over streaming fragments.
    for chunk in reversed(chunks):
        if chunk.get("object") != "chat.completion.chunk":
            return chunk

    # Merge streaming deltas into one synthetic envelope.
    if chunks:
        merged: List[str] = []
        for chunk in chunks:
            choices = chunk.get("choices") or []
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        merged.append(content)
        if merged:
            return {"choices": [{"message": {"content": "".join(merged)}}]}
        # Final chunk only (usage trailer): fall through to raw_decode.

    # Leading JSON object with trailing junk (`{...}data: [DONE]`).
    try:
        parsed, _ = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "openai_compatible returned an unparseable body"
        ) from exc
    if isinstance(parsed, dict):
        return parsed
    raise RuntimeError("openai_compatible returned a non-object body")


def _extract_message_text(body: Dict[str, Any]) -> str:
    """Pull the assistant text out of a chat-completions envelope.

    Handles `message.content` as a string, as a content-part list, and the
    legacy `choices[].text` shape. Raises RuntimeError when no text exists.
    """
    choices = body.get("choices") or []
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("openai_compatible response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("openai_compatible response choice is not an object")

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("text")
            ]
            if parts:
                return "".join(parts)

    text = first.get("text")
    if isinstance(text, str) and text.strip():
        return text

    raise RuntimeError("openai_compatible response missing choices[0].message.content")


class OpenAICompatibleProvider(VLMProvider):
    name = "openai_compatible"

    def __init__(self) -> None:
        self._key = settings.vlm_api_key
        self._base = (settings.vlm_api_base_url or "").rstrip("/")
        self._model = settings.vlm_openai_model or ""
        self._client = get_shared_httpx_client() if (self._key and self._base) else None

    def is_configured(self) -> bool:
        return bool(self._key) and bool(self._base) and bool(self._model)

    @property
    def model(self) -> str:
        return self._model or "<VLM_OPENAI_MODEL not set>"

    def with_model(self, model: str) -> "OpenAICompatibleProvider":
        clone = object.__new__(OpenAICompatibleProvider)
        clone._key = self._key
        clone._base = self._base
        clone._client = self._client
        clone._model = model
        return clone

    async def connectivity_check(self) -> None:
        if not self.is_configured():
            raise RuntimeError(
                "openai_compatible requires VLM_API_BASE_URL, VLM_API_KEY and VLM_OPENAI_MODEL"
            )
        response = await self._client.get(
            f"{self._base}/models", headers=self._headers()
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"openai_compatible /models returned HTTP {response.status_code}"
            )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }

    async def _call_once(self, selfie_b64: str, dl_b64: str, repair_text: str = "") -> str:
        if not self.is_configured():
            raise ValueError(
                "openai_compatible requires VLM_API_BASE_URL, VLM_API_KEY and VLM_OPENAI_MODEL"
            )

        text_block = (
            "Image 1 is the live selfie crop, Image 2 is the ID portrait face crop. "
            "Analyse biometric match and liveness. Reply with JSON only."
        )
        if repair_text:
            text_block += f"\n\nYour previous reply was:\n{repair_text[:settings.vlm_repair_context_chars]}"

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": BIOMETRIC_SYSTEM_PROMPT + "\n\n" + _REPAIR_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{selfie_b64}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{dl_b64}"},
                        },
                        {"type": "text", "text": text_block},
                    ],
                },
            ],
            "temperature": settings.vlm_temperature,
            "max_tokens": settings.vlm_max_tokens,
        }
        if settings.vlm_use_response_format:
            payload["response_format"] = {"type": "json_object"}

        response = await self._client.post(
            f"{self._base}/chat/completions",
            headers=self._headers(),
            json=payload,
        )
        if response.status_code == 400 and "response_format" in response.text.lower() and "response_format" in payload:
            logger.warning("Gateway rejected response_format, retrying without response_format...")
            payload.pop("response_format", None)
            response = await self._client.post(
                f"{self._base}/chat/completions",
                headers=self._headers(),
                json=payload,
            )

        if response.status_code >= 400:
            raise RuntimeError(
                f"openai_compatible /chat/completions returned HTTP {response.status_code}: {response.text[:300]}"
            )
        body = _parse_chat_body(response.text)
        return _extract_message_text(body)


# --------------------------------------------------------------------------
# Mock (tests / local staging only)
# --------------------------------------------------------------------------


class MockProvider(VLMProvider):
    name = "mock"

    def is_configured(self) -> bool:
        if settings.environment.lower() in ("production", "prod"):
            raise RuntimeError(
                "VLM_PROVIDER=mock is refused in production. Use a real cloud provider."
            )
        return True

    @property
    def model(self) -> str:
        return "mock"

    async def connectivity_check(self) -> None:
        return None

    async def _call_once(self, selfie_b64: str, dl_b64: str, repair_text: str = "") -> str:
        return json.dumps(
            {
                "is_match": settings.mock_vlm_match,
                "confidence_score": settings.mock_vlm_confidence,
                "is_live_selfie": settings.mock_vlm_liveness,
                "estimated_age_delta_years": 0,
                "facial_feature_notes": "Deterministic mock provider response.",
                "reasoning": "Mock provider: no cloud inference was performed.",
                "verdict": "MATCH_CONFIRMED" if settings.mock_vlm_match else "MISMATCH",
            }
        )


_PROVIDER_REGISTRY = {
    "openai_compatible": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
    "mock": MockProvider,
}


def build_provider(name: str) -> VLMProvider:
    factory = _PROVIDER_REGISTRY.get(name)
    if factory is None:
        raise ValueError(f"Unknown VLM provider '{name}'")
    return factory()


def build_model_candidates() -> List[VLMProvider]:
    """
    Build the ordered list of concrete providers to try.

    A provider that supports `with_model` contributes one entry per configured
    model fallback, so a bad model ID is skipped individually rather than
    poisoning the whole provider.
    """
    candidates: List[VLMProvider] = []

    for provider_name in settings.provider_fallback_list:
        try:
            provider = build_provider(provider_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cannot build VLM provider '%s': %s", provider_name, type(exc).__name__)
            continue

        if not provider.is_configured():
            continue

        with_model = getattr(provider, "with_model", None)
        if callable(with_model) and settings.model_fallback_list:
            models = [provider.model] + [
                m for m in settings.model_fallback_list if m != provider.model
            ]
            for model in models:
                candidates.append(with_model(model))
        else:
            candidates.append(provider)

    return candidates


__all__ = [
    "BiometricVerdictValidationError",
    "ProviderResult",
    "VLMProvider",
    "AnthropicProvider",
    "OpenAICompatibleProvider",
    "MockProvider",
    "build_provider",
    "build_model_candidates",
    "classify_error",
    "get_shared_httpx_client",
    "close_shared_httpx_client",
    "BIOMETRIC_SYSTEM_PROMPT",
]

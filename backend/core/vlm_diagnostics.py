"""
Phase 0: VLM diagnostics.

Provides three things the service previously lacked:
  1. Secret masking so keys never leak into logs, telemetry, or API responses.
  2. An offline readiness probe (key presence + key-shape validation + model IDs)
     so a misconfigured provider is visible without spending a network call.
  3. Failure classification, so "VLM not configured" is replaced by an actionable
     reason such as invalid_credentials / quota_exhausted / provider_unreachable.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.config import env_source_report, settings

logger = logging.getLogger(__name__)

# Reason codes surfaced in degraded_subsystems[].reason
REASON_NO_CREDENTIALS = "no_credentials"
REASON_INVALID_CREDENTIALS = "invalid_credentials"
REASON_QUOTA_EXHAUSTED = "quota_exhausted"
REASON_MODEL_UNAVAILABLE = "model_unavailable"
REASON_MODEL_OVERLOADED = "model_overloaded"
REASON_PROVIDER_UNREACHABLE = "provider_unreachable"
REASON_TIMEOUT = "timeout"
REASON_UNPARSEABLE_RESPONSE = "unparseable_response"
REASON_SDK_MISSING = "sdk_missing"
REASON_CONFIG_ERROR = "config_error"
REASON_PROVIDER_ERROR = "provider_error"

#: Reason codes worth retrying on a different provider rather than another model
#: of the same provider. A capacity error is a property of the account/region,
#: not of one model ID, so switching models of the same provider rarely helps.
CAPACITY_REASONS = frozenset({REASON_MODEL_OVERLOADED, REASON_QUOTA_EXHAUSTED})

#: Reason codes that mean the credential itself is wrong, so the remaining
#: providers will fail the same way.
CREDENTIAL_REASONS = frozenset({REASON_INVALID_CREDENTIALS, REASON_NO_CREDENTIALS})

#: Ordered low -> high confidence key-shape validators per provider.
KEY_SHAPE_HINTS: Dict[str, Tuple[str, ...]] = {
    "anthropic": ("sk-ant-",),
    "openai_compatible": ("sk-", "AQ.", "gsk_"),
}


def mask_secret(value: Optional[str]) -> Optional[str]:
    """Return a masked representation of a credential, or None when absent.

    Never returns enough of the secret to be usable. Applied to every log line,
    telemetry detail, and error message that could otherwise carry a key.
    """
    if not value:
        return None
    raw = str(value)
    if len(raw) <= 8:
        return f"***({len(raw)} chars)"
    return f"{raw[:4]}...{raw[-4:]} ({len(raw)} chars)"


def classify_error(exc: BaseException) -> str:
    """
    Map an arbitrary provider exception onto an actionable reason code.

    Order matters: capacity/overload is checked before generic 5xx handling so a
    transient "503 UNAVAILABLE - this model is currently experiencing high demand"
    is reported as a retryable `model_overloaded` rather than a vague
    `provider_error`.
    """
    name = type(exc).__name__.lower()
    text = str(exc).lower()

    # Structured status/code attributes are more reliable than message text.
    status = str(getattr(exc, "status", "") or "").upper()
    code = getattr(exc, "code", None)
    code = str(code) if code is not None else ""

    if "timeout" in name or "timeout" in text:
        return REASON_TIMEOUT
    if isinstance(exc, asyncio.TimeoutError):
        return REASON_TIMEOUT

    # A locally-raised "not configured" must be classified before the api_key
    # check, otherwise ValueError("VLM_API_KEY is not configured") is
    # misreported as invalid_credentials.
    if isinstance(exc, (ValueError, TypeError, KeyError)) and (
        "not configured" in text or "not set" in text or "no credentials" in text
    ):
        return REASON_CONFIG_ERROR

    # Capacity / overload: 429, 503, RESOURCE_EXHAUSTED, "high demand",
    # "overloaded", "capacity". Retryable on a different provider.
    if (
        "resource_exhausted" in text
        or "high demand" in text
        or "overloaded" in text
        or "capacity" in text
        or "rate limit" in text
        or "429" in text
        or "503" in text
        or code == "429"
        or code == "503"
        or status in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "OVERLOADED")
    ):
        # Billing/quota exhaustion is distinct from transient rate limiting:
        # only the former will not clear on its own.
        if "quota" in text or "billing" in text or "exceeded your current quota" in text:
            return REASON_QUOTA_EXHAUSTED
        return REASON_MODEL_OVERLOADED

    if "not found" in text or "not_found" in text or "404" in text or code == "404":
        return REASON_MODEL_UNAVAILABLE
    if "deprecat" in text or "retired" in text or "no longer" in text:
        return REASON_MODEL_UNAVAILABLE
    if (
        "api key" in text
        or "api_key" in text
        or "unauthenticated" in text
        or "permission" in text
        or "credential" in text
        or "401" in text
        or "403" in text
        or code in ("401", "403")
        or status in ("UNAUTHENTICATED", "PERMISSION_DENIED")
    ):
        return REASON_INVALID_CREDENTIALS
    if (
        "connect" in text
        or "network" in text
        or "dns" in text
        or "unreachable" in text
        or "ssl" in text
        or "502" in text
        or "504" in text
    ):
        return REASON_PROVIDER_UNREACHABLE
    if isinstance(exc, ImportError):
        return REASON_SDK_MISSING
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return REASON_CONFIG_ERROR
    return REASON_PROVIDER_ERROR


def _validate_key_shape(provider: str, key: Optional[str]) -> Tuple[bool, str]:
    """
    Return (shape_ok, human_hint).

    This is an ADVISORY pre-flight hint only. A key that does not match the
    expected prefix may still be a valid credential behind a corporate proxy or
    gateway, so a shape mismatch must never be treated as proof of failure once
    a provider has actually succeeded (see VLMDiagnostics.verified_providers).
    """
    if not key:
        return False, "no key present"
    hints = KEY_SHAPE_HINTS.get(provider)
    if not hints:
        return True, "no shape constraint for this provider"
    if any(key.startswith(prefix) for prefix in hints):
        return True, f"matches expected prefix ({' or '.join(hints)})"
    return False, (
        f"key does not start with any of {' / '.join(hints)}; got a {len(key)}-char "
        f"value starting with {key[:2]!r}. This is a common cause of a silent "
        "SERVICE_UNAVAILABLE verdict, but it is only a hint: gateways and proxies "
        "legitimately issue keys of other shapes. Confirm with ?live=true."
    )


def _sdk_importable(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _resolve_credentials(provider: str) -> Tuple[Optional[str], str]:
    """Return (key, source_description) for a provider without leaking the key."""
    if provider == "anthropic":
        key = settings.anthropic_api_key
        if key:
            return key, "settings.anthropic_api_key (ANTHROPIC_API_KEY)"
        return None, "unset"
    if provider == "openai_compatible":
        key = settings.vlm_api_key
        if key:
            return key, "settings.vlm_api_key (VLM_API_KEY)"
        return None, "unset"
    if provider == "mock":
        return "mock", "built-in (no credential required)"
    return None, "unknown provider"


def _resolve_model(provider: str) -> str:
    if provider == "anthropic":
        return settings.anthropic_model
    if provider == "openai_compatible":
        return settings.vlm_openai_model or "<VLM_OPENAI_MODEL not set>"
    if provider == "mock":
        return "mock"
    return "<unknown>"


def probe_provider(provider: str) -> Dict[str, Any]:
    """Offline readiness probe. Performs no network I/O."""
    key, source = _resolve_credentials(provider)
    shape_ok, shape_hint = _validate_key_shape(provider, key if key != "mock" else None)

    sdk_module = {
        "anthropic": "anthropic",
        "openai_compatible": "httpx",
        "mock": None,
    }.get(provider)

    sdk_ok = True if sdk_module is None else _sdk_importable(sdk_module)

    base_url = None
    if provider == "openai_compatible":
        base_url = settings.vlm_api_base_url

    blocking: List[str] = []
    if provider != "mock" and not key:
        blocking.append(REASON_NO_CREDENTIALS)
    if provider == "openai_compatible" and not base_url:
        blocking.append(REASON_CONFIG_ERROR)
    if not sdk_ok:
        blocking.append(REASON_SDK_MISSING)

    return {
        "provider": provider,
        "configured": bool(key) if provider != "mock" else True,
        "credential_source": source,
        "key_masked": mask_secret(key) if key != "mock" else "mock",
        "key_shape_valid": shape_ok,
        "key_shape_hint": shape_hint,
        "sdk_importable": sdk_ok,
        "base_url": base_url,
        "model": _resolve_model(provider),
        "ready": len(blocking) == 0,
        "blocking_reason": blocking[0] if blocking else None,
    }


class VLMDiagnostics:
    """Aggregated VLM readiness + live-probe state for the diagnostics endpoint."""

    def __init__(self) -> None:
        self._last_error: Optional[Dict[str, Any]] = None
        #: Providers observed to return a valid verdict at least once. Real
        #: traffic overrides the advisory key-shape heuristic, so a working
        #: gateway key is never reported as a misconfiguration.
        self.verified_providers: set = set()

    def record_success(self, provider: str) -> None:
        """Mark a provider as proven working; suppresses its shape warning."""
        if provider in self.verified_providers:
            return
        self.verified_providers.add(provider)
        logger.info("VLM provider '%s' returned a valid verdict; treating its credential as verified.", provider)

    def probe_all(self) -> Dict[str, Any]:
        chain = settings.provider_fallback_list
        providers = [probe_provider(p) for p in chain]

        usable = [p for p in providers if p["ready"]]
        configured_but_bad_shape = [
            p
            for p in providers
            if p["configured"]
            and not p["key_shape_valid"]
            and p["provider"] not in self.verified_providers
        ]

        return {
            "primary_provider": settings.vlm_provider,
            "provider_fallback_chain": chain,
            "model_fallbacks": settings.model_fallback_list,
            "vlm_ready": len(usable) > 0,
            "effective_provider": usable[0]["provider"] if usable else None,
            "usable_providers": [p["provider"] for p in usable],
            "providers": providers,
            "key_shape_mismatches": [p["provider"] for p in configured_but_bad_shape],
            "verified_providers": sorted(self.verified_providers),
            "timeout_budget": {
                "pipeline_timeout_seconds": settings.pipeline_timeout_seconds,
                "vlm_total_timeout_seconds": settings.vlm_total_timeout_seconds,
                "vlm_call_timeout_seconds": settings.vlm_call_timeout_seconds,
            },
            "env_sources": env_source_report.to_dict(),
            "last_error": self._last_error,
        }

    def record_error(self, provider: str, model: Optional[str], exc: BaseException) -> str:
        reason = classify_error(exc)
        self._last_error = {
            "provider": provider,
            "model": model,
            "reason": reason,
            "error_class": type(exc).__name__,
            "detail": _safe_detail(exc),
        }
        logger.warning(
            "VLM provider '%s' failed (%s / %s): %s",
            provider,
            reason,
            type(exc).__name__,
            _safe_detail(exc),
        )
        return reason

    def record_unparseable(self, provider: str, model: Optional[str], detail: str) -> None:
        self._last_error = {
            "provider": provider,
            "model": model,
            "reason": REASON_UNPARSEABLE_RESPONSE,
            "error_class": "BiometricVerdictValidationError",
            "detail": detail[:300],
        }

    async def live_probe(self, timeout: float = 8.0) -> Dict[str, Any]:
        """Send one minimal text-only call per usable provider. No images are sent."""
        from core.vlm_providers import build_provider

        results: List[Dict[str, Any]] = []
        for provider_name in settings.provider_fallback_list:
            probe = probe_provider(provider_name)
            if not probe["ready"]:
                results.append({**probe, "live": None, "live_detail": "skipped: not ready"})
                continue

            provider = build_provider(provider_name)
            try:
                async with asyncio.timeout(timeout):
                    await provider.connectivity_check()
                self.record_success(provider_name)
                results.append({**probe, "live": True, "live_detail": "reachable"})
            except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
                reason = self.record_error(provider_name, probe.get("model"), exc)
                results.append(
                    {**probe, "live": False, "live_detail": f"{reason}: {type(exc).__name__}"}
                )

        return {
            "attempted": True,
            "timeout_seconds": timeout,
            "results": results,
        }


def _safe_detail(exc: BaseException) -> str:
    """Exception text with any credential-shaped substring masked out."""
    detail = str(exc) or type(exc).__name__
    try:
        for candidate in (
            settings.anthropic_api_key,
            settings.vlm_api_key,
            settings.api_secret_key,
        ):
            if candidate and len(candidate) > 8 and candidate in detail:
                detail = detail.replace(candidate, mask_secret(candidate) or "***")
    except Exception:  # noqa: BLE001 - masking must never raise
        pass
    return detail[:300]


vlm_diagnostics = VLMDiagnostics()

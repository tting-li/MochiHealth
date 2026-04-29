"""Ollama HTTP helpers with timeout and retries."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from src.agent_debug import agent_debug
from src.config import (
    MAX_RETRIES,
    OLLAMA_AUTO_FALLBACK,
    OLLAMA_BASE_URL,
    OLLAMA_FALLBACK_MODEL,
    REQUEST_TIMEOUT_S,
)

logger = logging.getLogger(__name__)


def _tensor_blob_error(msg: str) -> bool:
    return "wrong number of tensors" in msg or "done_getting_tensors" in msg


def _post_once(
    model: str,
    prompt: str,
    system: str | None,
    temperature: float,
    base_url: str | None,
) -> str:
    """Single POST; raises RuntimeError on HTTP/model errors (do not transport-retry)."""
    url = (base_url or OLLAMA_BASE_URL).rstrip("/") + "/api/generate"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system

    # #region agent log
    agent_debug(
        "H1-H2-H3",
        "src/ollama_client.py:_post_once:request",
        "posting_generate",
        {
            "model": model,
            "temperature": temperature,
            "prompt_chars": len(prompt or ""),
            "system_chars": len(system or ""),
            "has_system": bool(system),
        },
    )
    # #endregion agent log

    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_S)

    # #region agent log
    agent_debug(
        "H1-H2-H3",
        "src/ollama_client.py:_post_once:response",
        "got_response",
        {
            "model": model,
            "status_code": r.status_code,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "content_type": r.headers.get("content-type", ""),
            "body_snippet": (r.text or "")[:500],
        },
    )
    # #endregion agent log

    if r.status_code >= 400:
        err_msg: str | None = None
        try:
            err_msg = (r.json() or {}).get("error")
        except Exception:
            err_msg = None
        if not isinstance(err_msg, str):
            err_msg = (r.text or "")[:500] or None

        # #region agent log
        agent_debug(
            "H1",
            "src/ollama_client.py:_post_once:ollama_error",
            "ollama_error_body",
            {
                "model": model,
                "status_code": r.status_code,
                "error": (err_msg or "")[:400],
            },
        )
        # #endregion agent log

        logger.error(
            "ollama_generate_http_error status=%s model=%s error=%s",
            r.status_code,
            model,
            (err_msg or "")[:300],
        )

        hint = ""
        if err_msg and _tensor_blob_error(err_msg):
            hint = (
                f" Known-bad or incompatible blob for `{model}` with this Ollama build. "
                f"Try `ollama pull {OLLAMA_FALLBACK_MODEL}` (fallback) or upgrade Ollama."
            )
        elif r.status_code >= 500:
            hint = " Try restarting the Ollama app or choosing another model in the sidebar."

        raise RuntimeError(
            f"Ollama returned HTTP {r.status_code} for model `{model}`: "
            f"{err_msg or r.reason}{hint}"
        )

    data = r.json()
    return str(data.get("response", ""))


def _post_with_transport_retries(
    model: str,
    prompt: str,
    system: str | None,
    temperature: float,
    base_url: str | None,
) -> str:
    """Retry only on transient network failures."""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return _post_once(model, prompt, system, temperature, base_url)
        except RuntimeError:
            raise
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            logger.warning("ollama transport attempt %s failed: %s", attempt, e)
            # #region agent log
            agent_debug(
                "H4",
                "src/ollama_client.py:_post_with_transport_retries:transport",
                "transport_error",
                {"attempt": attempt, "exc_type": type(e).__name__, "exc_str": str(e)[:300]},
            )
            # #endregion agent log
            if attempt <= MAX_RETRIES:
                time.sleep(0.5 * attempt)
    assert last_err is not None
    raise last_err


def ollama_generate(
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.1,
    base_url: str | None = None,
) -> str:
    """Call Ollama /api/generate (non-streaming). Returns full response text."""
    try:
        return _post_with_transport_retries(
            model, prompt, system, temperature, base_url
        )
    except RuntimeError as e:
        primary_msg = str(e)
        fb = (OLLAMA_FALLBACK_MODEL or "").strip()
        if (
            OLLAMA_AUTO_FALLBACK
            and fb
            and fb != model.strip()
            and _tensor_blob_error(primary_msg)
        ):
            # #region agent log
            agent_debug(
                "H7",
                "src/ollama_client.py:ollama_generate:fallback",
                "tensor_error_try_fallback",
                {"primary_model": model, "fallback_model": fb},
            )
            # #endregion agent log
            logger.warning(
                "ollama tensor load error on primary model=%s; trying fallback=%s",
                model,
                fb,
            )
            try:
                return _post_with_transport_retries(
                    fb, prompt, system, temperature, base_url
                )
            except RuntimeError as e2:
                raise RuntimeError(
                    f"Primary model `{model}` failed:\n{primary_msg}\n\n"
                    f"Automatic fallback model `{fb}` failed:\n{e2}\n\n"
                    f"Install the fallback with `ollama pull {fb}`, or set OLLAMA_MODEL / "
                    "OLLAMA_FALLBACK_MODEL in `.env`, or upgrade Ollama."
                ) from e2
        raise


def ollama_health(model: str, base_url: str | None = None) -> bool:
    url = (base_url or OLLAMA_BASE_URL).rstrip("/") + "/api/tags"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        names = [t.get("name", "") for t in r.json().get("models", [])]
        if not names:
            return False
        m = model.strip()
        if m in names:
            return True
        base = m.split(":")[0]
        return any((n.split(":")[0] if ":" in n else n) == base for n in names)
    except Exception:
        return False

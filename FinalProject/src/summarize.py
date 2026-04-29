"""LLM-based session summarization for the journal (title + summary + triage tags)."""

from __future__ import annotations

import json
import re
from typing import Any

from src.ollama_client import ollama_generate

TRIAGE_TAGS = ["routine", "monitor", "urgent", "emergency"]

JSON_RE = re.compile(r"\{[\s\S]*\}")


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # Drop leading fence line, keep body.
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_json_object(raw: str) -> str | None:
    raw = _strip_code_fences(raw)
    m = JSON_RE.search(raw)
    if not m:
        return None
    return m.group(0).strip()


def _safe_json_loads(s: str) -> dict[str, Any]:
    """
    Best-effort JSON parsing for small model outputs.
    Handles common issues like trailing commas and smart quotes.
    """
    s = s.strip()
    # Replace smart quotes.
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    # Remove trailing commas before } or ].
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return json.loads(s)


def build_transcript(messages: list[dict], max_turns: int = 16) -> str:
    tail = messages[-(max_turns * 2) :]
    lines = []
    for m in tail:
        role = (m.get("role") or "").upper()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        # Strip HTML badges if present.
        content = re.sub(r"<[^>]+>", "", content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def summarize_session(
    messages: list[dict],
    *,
    model: str,
) -> dict[str, Any]:
    transcript = build_transcript(messages)
    attempts: list[tuple[str, str]] = []

    system1 = (
        "You are summarizing a health chat journal for the user. "
        "Do NOT add medical advice. Just summarize what was discussed.\n\n"
        "Return STRICT JSON only with keys: title, summary, triage_tags.\n"
        f"triage_tags must be a subset of {TRIAGE_TAGS}.\n"
        "summary should be 2-5 bullet-like sentences in plain text.\n"
        "Do NOT wrap the JSON in markdown.\n"
    )
    prompt1 = (
        "Summarize this transcript for the user's journal.\n\n"
        f"{transcript}\n\n"
        "JSON:"
    )
    attempts.append((system1, prompt1))

    # Retry with an explicit JSON skeleton if the first attempt isn't parseable.
    system2 = (
        "Return ONLY valid JSON. No prose.\n"
        f"triage_tags must be a JSON list subset of {TRIAGE_TAGS}.\n"
    )
    prompt2 = (
        "Fill this JSON template based on the transcript.\n\n"
        "Template:\n"
        '{ "title": "", "summary": "", "triage_tags": [] }\n\n'
        f"Transcript:\n{transcript}\n\n"
        "JSON:"
    )
    attempts.append((system2, prompt2))

    last_raw = ""
    obj: dict[str, Any] | None = None
    for system, prompt in attempts:
        last_raw = ollama_generate(model=model, prompt=prompt, system=system, temperature=0.0)
        js = _extract_json_object(last_raw)
        if not js:
            continue
        try:
            obj = _safe_json_loads(js)
            break
        except Exception:
            obj = None
            continue

    if obj is None:
        # Deterministic fallback: never block the UI if the model output is messy.
        # Title: first user message; Summary: first few lines of transcript.
        title_guess = ""
        for m in messages:
            if m.get("role") == "user" and m.get("content"):
                title_guess = str(m["content"]).strip()
                break
        title_guess = (title_guess[:48] + ("…" if len(title_guess) > 48 else "")) if title_guess else "Conversation"
        short = transcript.strip().splitlines()[:6]
        summary_guess = " ".join([re.sub(r"^(USER|ASSISTANT):\s*", "", ln).strip() for ln in short]).strip()
        if len(summary_guess) > 400:
            summary_guess = summary_guess[:400].rstrip() + "…"
        return {"title": title_guess, "summary": summary_guess, "triage_tags": []}

    title = str(obj.get("title", "")).strip()
    summary = str(obj.get("summary", "")).strip()
    tags = obj.get("triage_tags") or []
    if not isinstance(tags, list):
        tags = []
    tags = [t for t in (str(x).strip().lower() for x in tags) if t in TRIAGE_TAGS]
    return {"title": title, "summary": summary, "triage_tags": sorted(set(tags))}


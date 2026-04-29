"""Session identifiers and rate limiting helpers."""

from __future__ import annotations

import time
import uuid
from collections import deque


def new_session_id() -> str:
    return str(uuid.uuid4())


def check_rate_limit(
    timestamps: deque[float],
    now: float | None = None,
    *,
    max_messages: int = 20,
    window_seconds: float = 60.0,
) -> tuple[bool, str | None]:
    """
    Sliding-window limit on message count.
    timestamps: deque of recent message times (mutated in place on record).
    Returns (allowed, error_message).
    """
    now = now or time.monotonic()
    while timestamps and now - timestamps[0] > window_seconds:
        timestamps.popleft()
    if len(timestamps) >= max_messages:
        return (
            False,
            f"Rate limit: at most {max_messages} messages per {int(window_seconds)} seconds. Please wait a moment.",
        )
    return True, None


def record_message_time(timestamps: deque[float], now: float | None = None) -> None:
    now = now or time.monotonic()
    timestamps.append(now)

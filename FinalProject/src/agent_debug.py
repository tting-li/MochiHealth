"""NDJSON debug lines for Cursor debug mode (dual path: .cursor + logs/)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_LOG_PATHS = (
    _ROOT / ".cursor" / "debug-998ffe.log",
    _ROOT / "logs" / "debug-998ffe.ndjson",
)
_SESSION_ID = "998ffe"


def agent_debug(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
    *,
    run_id: str = "pre-fix",
) -> None:
    payload = {
        "sessionId": _SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    for p in _LOG_PATHS:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

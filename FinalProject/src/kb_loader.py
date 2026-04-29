"""Load medical_kb.json into LangChain Documents."""

from __future__ import annotations

import json
from pathlib import Path
from langchain_core.documents import Document

from src.config import MEDICAL_KB_PATH


def load_medical_documents(kb_path: Path | None = None) -> list[Document]:
    path = kb_path or MEDICAL_KB_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    docs: list[Document] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        red = row.get("red_flags") or []
        red_text = "; ".join(red) if isinstance(red, list) else str(red)
        page = (
            f"Title: {row.get('title', '')}\n\n"
            f"{row.get('content', '')}\n\n"
            f"Red flags to watch for: {red_text}"
        )
        meta = {
            "id": row.get("id", ""),
            "title": row.get("title", ""),
            "triage_level": row.get("triage_level", ""),
            "source": row.get("source", "MedlinePlus"),
        }
        docs.append(Document(page_content=page, metadata=meta))
    return docs

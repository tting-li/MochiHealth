"""In-memory vector retrieval + prompt assembly + Ollama generation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from src.config import (
    DEFAULT_RETRIEVAL_K,
    EMBEDDING_MODEL_NAME,
    MEDICAL_KB_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    VECTOR_CACHE_DIR,
)
from src.kb_loader import load_medical_documents
from src.ollama_client import ollama_generate
from src.prompts import PROMPT_VARIANTS, build_user_message, format_context

logger = logging.getLogger(__name__)

STRUCTURED_RE = re.compile(
    r"###STRUCTURED_OUTPUT###\s*(\{.*\})\s*$", re.DOTALL | re.MULTILINE
)

VALID_TRIAGE = {"routine", "monitor", "urgent", "emergency"}


_emb_singleton = None


def get_embeddings():
    """Lazy singleton (avoid reloading ST model on every retrieval)."""
    global _emb_singleton
    if _emb_singleton is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        _emb_singleton = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _emb_singleton


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


class NumpyVectorStore:
    """Cosine similarity over precomputed document embeddings (no FAISS)."""

    def __init__(self, docs: list[Document], vectors: np.ndarray):
        self.docs = docs
        self.vectors = _normalize(np.asarray(vectors, dtype=np.float32))

    def similarity_search(self, query: str, k: int = DEFAULT_RETRIEVAL_K) -> list[Document]:
        emb = get_embeddings()
        qv = np.asarray(emb.embed_query(query), dtype=np.float32)
        qv = _normalize(qv.reshape(1, -1))
        scores = (self.vectors @ qv.T).ravel()
        k = min(k, len(self.docs))
        idx = np.argsort(-scores)[:k]
        return [self.docs[int(i)] for i in idx]


def build_vectorstore(kb_path: Path | None = None, use_cache: bool = True) -> NumpyVectorStore:
    kb_path = kb_path or MEDICAL_KB_PATH
    VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_vecs = VECTOR_CACHE_DIR / "doc_vectors.npy"
    cache_meta = VECTOR_CACHE_DIR / "cache_meta.json"

    docs = load_medical_documents(kb_path)
    mtime = kb_path.stat().st_mtime

    if use_cache and cache_vecs.exists() and cache_meta.exists():
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            if meta.get("kb_mtime") == mtime:
                vecs = np.load(cache_vecs)
                return NumpyVectorStore(docs, vecs)
        except Exception as e:
            logger.warning("embedding cache load failed, rebuilding: %s", e)

    emb = get_embeddings()
    texts = [d.page_content for d in docs]
    vecs = np.asarray(emb.embed_documents(texts), dtype=np.float32)
    if use_cache:
        np.save(cache_vecs, vecs)
        cache_meta.write_text(json.dumps({"kb_mtime": mtime}), encoding="utf-8")
    return NumpyVectorStore(docs, vecs)


def retrieve(vs: NumpyVectorStore, query: str, k: int = DEFAULT_RETRIEVAL_K) -> list[Document]:
    return vs.similarity_search(query, k=k)


@dataclass
class TurnResult:
    answer_text: str
    triage_level: str | None
    symptoms_this_turn: list[str]
    retrieved: list[Document]
    raw_response: str


def parse_structured_output(raw: str) -> tuple[str | None, list[str], str]:
    """Return (triage_level, symptoms, display_text without trailing marker block)."""
    m = STRUCTURED_RE.search(raw)
    if not m:
        return None, [], raw.strip()
    json_str = m.group(1)
    display = raw[: m.start()].strip()
    try:
        obj = json.loads(json_str)
        tri = obj.get("triage_level")
        if isinstance(tri, str) and tri.lower() in VALID_TRIAGE:
            tri_out = tri.lower()
        else:
            tri_out = None
        sym = obj.get("symptoms_this_turn") or []
        if not isinstance(sym, list):
            sym = []
        sym = [str(s).strip() for s in sym if str(s).strip()]
        return tri_out, sym, display
    except json.JSONDecodeError:
        return None, [], display


def format_history_block(messages: list[dict], max_turns: int = 6) -> str:
    """Last N user/assistant pairs as plain text."""
    tail = messages[-(max_turns * 2) :]
    lines = []
    for m in tail:
        role = m.get("role", "")
        content = m.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines) if lines else "(No prior turns.)"


def run_turn(
    vs: NumpyVectorStore,
    user_text: str,
    messages: list[dict],
    prompt_variant: str,
    model: str = OLLAMA_MODEL,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    temperature: float = 0.1,
    base_url: str | None = None,
) -> TurnResult:
    docs = retrieve(vs, user_text, k=retrieval_k)
    context = format_context(docs)
    history_block = format_history_block(messages)
    system = PROMPT_VARIANTS.get(prompt_variant, PROMPT_VARIANTS["zero_shot"])
    user_msg = build_user_message(user_text, context, history_block)

    raw = ollama_generate(
        model=model,
        prompt=user_msg,
        system=system,
        temperature=temperature,
        base_url=base_url or OLLAMA_BASE_URL,
    )
    tri, sym, display = parse_structured_output(raw)
    if not display:
        display = raw.strip()
    return TurnResult(
        answer_text=display,
        triage_level=tri,
        symptoms_this_turn=sym,
        retrieved=docs,
        raw_response=raw,
    )


def coherence_score(embedding_model, answer: str, sources: list[Document]) -> float:
    """Cosine similarity between answer and concatenated source text."""
    if not answer.strip():
        return 0.0
    src = "\n".join(d.page_content for d in sources)
    if not src.strip():
        return 0.0
    a = np.asarray(embedding_model.embed_query(answer[:8000]), dtype=np.float32)
    b = np.asarray(embedding_model.embed_query(src[:8000]), dtype=np.float32)
    a = _normalize(a.reshape(1, -1)).ravel()
    b = _normalize(b.reshape(1, -1)).ravel()
    return float(np.dot(a, b))

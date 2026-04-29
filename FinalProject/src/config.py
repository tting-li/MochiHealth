"""Paths and defaults for the health journal assistant."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"
# Use project-local HF cache so installs work in restricted environments / CI
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE_DIR))
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MEDICAL_KB_PATH = PROJECT_ROOT / "medical_kb.json"
EVAL_CASES_PATH = PROJECT_ROOT / "data" / "eval_cases.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"
VECTOR_CACHE_DIR = PROJECT_ROOT / "data" / "vector_cache"
DB_PATH = PROJECT_ROOT / "data" / "app.db"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
# If primary model hits the known llama.cpp tensor manifest bug, try this model once.
OLLAMA_FALLBACK_MODEL = os.environ.get("OLLAMA_FALLBACK_MODEL", "phi3:mini")
OLLAMA_AUTO_FALLBACK = os.environ.get("OLLAMA_AUTO_FALLBACK", "1").lower() in (
    "1",
    "true",
    "yes",
)
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
)

DEFAULT_RETRIEVAL_K = 4
REQUEST_TIMEOUT_S = 120
MAX_RETRIES = 2

# Rate limiting: max user messages per rolling window
RATE_LIMIT_MAX_MESSAGES = 20
RATE_LIMIT_WINDOW_SECONDS = 60

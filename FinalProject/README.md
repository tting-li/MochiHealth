# Mochi Health Personal Assistant (CS372 Final Project)

## What it does

This is an **educational prototype**: a Streamlit web app where you can describe symptoms in natural language to the app. The app retrieves **MedlinePlus-style** patient education passages from a small local knowledge base, then uses a **local Ollama** language model to produce **informational** guidance with citations to the sources it references. It supports **multi-turn** chat and tracks mentioned symptoms in session state.

## disclaimer to all users -- this does NOT replace any medical assistance!!!

**This is not medical advice, not a clinical diagnosis, and not a substitute for a licensed clinician or emergency services. If there is a medical emergency, please contact local emergency number -- 911 in the U.S.**

The model is more health-topic-oriented focused LLM than a generic chatbot as answers are trained using retrieved medical text plus explicit triage-style categories—but it **cannot replace** professional evaluation.

## Knowledge base provenance

The file [`medical_kb.json`](medical_kb.json) is a **generated / curated** JSON dataset (not a full crawl of MedlinePlus). Each record includes `title`, `content`, `triage_level`, `red_flags`, and `source` attribution.

This was currated for the class project purpose -- for further improvement, datasets from MedlinePlus or other resources would be used and cited to better match recent clinical guidance.

## Quick start

1. Install dependencies: see [SETUP.md](SETUP.md).
2. `ollama pull llama3.2:3b` and `ollama pull phi3:mini` (fallback avoids a rare Ollama tensor-load crash on some setups).
3. `streamlit run app.py` -- for frontend usage

## Video links

- **Demo video :** `https://www.loom.com/share/e4307fc82dfb4d69b06a11236167a726`
- **Technical walkthrough:** `https://duke.box.com/s/z3oeadep0id2jvwsxxf4jrihbaou4ymf`

## Evaluation

We compare three prompt strategies on a held-out labeled set in [`data/eval_cases.jsonl`](data/eval_cases.jsonl): **zero-shot**, **few-shot**, and **chain-of-thought**. Run `python scripts/eval_prompts.py` to regenerate tables and figures.

### Metrics

1. **Triage accuracy** — Exact match of parsed `triage_level` from the model against `gold_triage` (`routine`, `monitor`, `urgent`, `emergency`).
2. **Retrieval relevance** — Mean Precision@k (k = number of retrieved docs) using `gold_doc_ids`: fraction of retrieved docs that appear in the gold set (for single gold id, P@k is 1 if any retrieved doc matches).
3. **Response coherence** — Cosine similarity between the embedding of the assistant reply and the embedding of concatenated retrieved source text (higher = closer alignment with retrieved evidence).

### Generated artifacts

| Artifact                                                       | Description                                                                                     |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [results/prompt_comparison.csv](results/prompt_comparison.csv) | Per-example metrics by prompt variant; includes **`latency_ms`** (batch wall-clock per example) |
| [results/prompt_comparison.md](results/prompt_comparison.md)   | Markdown table + **offline latency** summary (mean / p50 / p95)                                 |
| [results/error_analysis.md](results/error_analysis.md)         | Qualitative + quantitative error discussion (zero-shot), edge/OOD rows, example failures        |
| [figures/confusion_matrix.png](figures/confusion_matrix.png)   | Confusion matrix (gold vs predicted) for best prompt variant on the test split                  |

**Limitations:** Small KB (curated subset), local small LM, and synthetic/heuristic labels for evaluation. Retrieval will miss out-of-domain queries; edge-case tests document refusals and safe fallbacks.

Run evaluation locally (uses `OLLAMA_MODEL`, default from [`src/config.py`](src/config.py); **`phi3:mini` recommended** if `llama3.2:3b` fails on your machine):

```bash
OLLAMA_MODEL=phi3:mini python scripts/eval_prompts.py
```

## Evidence map (self-assessment pointers)

Use this section (and [`docs/ml_rubric_claims.md`](docs/ml_rubric_claims.md)) when filling the CS 372 self-assessment: each bullet is a concrete place graders can inspect.

| Claim area                | Where to look                                                                                                                                                                                                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Modular structure         | [`src/`](src/) package split: RAG [`rag_chain.py`](src/rag_chain.py), prompts [`prompts.py`](src/prompts.py), Ollama client [`ollama_client.py`](src/ollama_client.py), journal/DB [`journal.py`](src/journal.py), [`db.py`](src/db.py), auth [`auth.py`](src/auth.py) |
| LangChain + RAG           | [`src/kb_loader.py`](src/kb_loader.py) (`Document` objects), [`src/rag_chain.py`](src/rag_chain.py) (retrieve → prompt → `ollama_generate`)                                                                                                                            |
| Multi-turn context        | [`src/rag_chain.py`](src/rag_chain.py) `format_history_block`, [`app.py`](app.py) chat state + SQLite persistence                                                                                                                                                      |
| Prompt variants + eval    | [`src/prompts.py`](src/prompts.py) `PROMPT_VARIANTS`; batch eval [`scripts/eval_prompts.py`](scripts/eval_prompts.py) → [`results/`](results/) + [`figures/`](figures/)                                                                                                |
| Inference time            | Interactive: [`app.py`](app.py) `logger.info("turn_ok ... ms=...")` and `logs/app.log`. Offline batch: **`latency_ms` column** and latency table in [`results/prompt_comparison.md`](results/prompt_comparison.md).                                                    |
| Production-ish safeguards | Rate limit [`src/session.py`](src/session.py) + [`app.py`](app.py); logging lines above; Ollama error handling [`src/ollama_client.py`](src/ollama_client.py)                                                                                                          |

## Project layout

- `app.py` — Streamlit UI
- `src/` — RAG chain, KB loader, Ollama client, prompts, session helpers
- `scripts/eval_prompts.py` — Batch evaluation
- `data/eval_cases.jsonl` — Labeled evaluation cases
- `medical_kb.json` — Curated knowledge JSON for RAG

## Model stack

- **LLM:** Ollama HTTP API (default `llama3.2:3b`; override with env).
- **Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers) via **LangChain** `HuggingFaceEmbeddings`; model weights cache under `.hf_cache/` in the project root (so the app works without writing to a global Hugging Face cache).
- **RAG:** Knowledge is loaded as LangChain `Document` objects with metadata; retrieval uses embedding cosine similarity (in-memory over the curated KB; optional numpy cache under `data/vector_cache/`).

## Logging policy

Logs go to [`logs/app.log`](logs/app.log). By design we log **session id**, **prompt variant**, **latency**, **retrieved document ids**, and **user message length** — not full user message text — to reduce sensitive health data in files. If you deploy publicly, review applicable privacy rules before changing this.

## Individual contributions

Solo project by Ting Ting Li

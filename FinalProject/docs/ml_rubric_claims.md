# Machine Learning rubric — claimed items (CS 372)

You may claim **up to 15** Machine Learning checklist items. Items below sum to **74** claimed ML points; course scoring **caps Machine Learning at 73**, so this set is intentionally slightly over-parameterized.

**Two slots are left unused** (fine per handout: fewer than 15 claims allowed).

| # | Rubric item (short) | Pts | Evidence pointer | Notes |
|---|---------------------|-----|------------------|-------|
| 1 | Modular code design | 3 | Package layout under [`src/`](../src/) (`rag_chain.py`, `prompts.py`, `ollama_client.py`, `journal.py`, `db.py`, `auth.py`) | Single cohesive split of concerns. |
| 2 | Sentence embeddings for retrieval | 5 | [`src/rag_chain.py`](../src/rag_chain.py) `get_embeddings()`, `NumpyVectorStore.similarity_search()` | Not claimed separately as generic “feature engineering” to avoid duplicate work. |
| 3 | Prompt engineering + ≥3 designs evaluated | 3 | [`scripts/eval_prompts.py`](../scripts/eval_prompts.py) `VARIANTS`; outputs [`results/prompt_comparison.md`](../results/prompt_comparison.md), [`results/prompt_comparison.csv`](../results/prompt_comparison.csv) | Compares zero-shot / few-shot / chain-of-thought. |
| 4 | Few-shot + chain-of-thought prompting | 5 | [`src/prompts.py`](../src/prompts.py) `PROMPT_VARIANTS` | Distinct system prompts including examples (few-shot) and CoT scaffold. |
| 5 | Multi-turn conversation + history | 7 | [`src/rag_chain.py`](../src/rag_chain.py) `format_history_block()`, `run_turn()`; [`app.py`](../app.py) `st.session_state.messages`, persistence via `ensure_session` / `append_message` | History folded into each prompt; UI keeps transcript + journal. |
| 6 | Transformer LM inference (local) | 3 | [`src/ollama_client.py`](../src/ollama_client.py) `ollama_generate()`; called from [`src/rag_chain.py`](../src/rag_chain.py) `run_turn()` | Local decoder LLM via Ollama HTTP API. |
| 7 | LangChain RAG “Used” tier | 5 | [`src/kb_loader.py`](../src/kb_loader.py) loads LangChain `Document`s; [`src/rag_chain.py`](../src/rag_chain.py) retrieval + generation | Framework-document pipeline + integration. |
| 8 | Deployed functional web app | 10 | [`app.py`](../app.py) Streamlit UI | Runnable locally via `streamlit run app.py`. |
| 9 | Production-oriented deployment (≥2) | 10 | Rate limiting [`src/session.py`](../src/session.py) + [`app.py`](../app.py); structured logging [`app.py`](../app.py) `logger.info("turn_ok"...)` + [`README.md`](../README.md) Logging policy; resilience [`src/ollama_client.py`](../src/ollama_client.py) | Multiple operational safeguards. |
| 10 | ≥3 evaluation metrics | 3 | [`README.md`](../README.md) Metrics; [`scripts/eval_prompts.py`](../scripts/eval_prompts.py) computes accuracy, P@k, coherence | Metrics tied to triage + retrieval + answer grounding. |
| 11 | Error analysis + visualization | 7 | [`scripts/eval_prompts.py`](../scripts/eval_prompts.py) confusion matrices → [`figures/`](../figures/); narrative [`results/error_analysis.md`](../results/error_analysis.md) | Confusion matrices + written failure discussion. |
| 12 | Inference time / efficiency | 3 | Per-turn ms in [`app.py`](../app.py); batch **latency_ms** in [`results/prompt_comparison.csv`](../results/prompt_comparison.csv) + summary in [`results/prompt_comparison.md`](../results/prompt_comparison.md) after eval script run | Operational + offline measurement. |
| 13 | Edge / OOD behavior | 5 | Edge categories in [`data/eval_cases.jsonl`](../data/eval_cases.jsonl); discussion in [`results/error_analysis.md`](../results/error_analysis.md) | Fiction / vague prompts exercised. |
| 14 | Qualitative **and** quantitative evaluation | 5 | Tables/CSV (quantitative) + prose interpretation (qualitative) in [`results/error_analysis.md`](../results/error_analysis.md) and Evaluation in [`README.md`](../README.md) | Paired numbers + interpretation. |
| 15 | Solo project credit | 10 | [`README.md`](../README.md) Individual contributions | Single author. |

**Duplicate-work rule:** Embedding retrieval is claimed once (row 2). Prompt-variant evaluation (row 3) is distinct from prompt text design (row 4).

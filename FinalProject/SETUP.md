# Setup

## Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running (`ollama serve` is usually automatic after install)
- Local chat models pulled. Defaults assume **both** of these exist (fallback avoids a known Ollama crash on some Macs):

```bash
ollama pull llama3.2:3b
ollama pull phi3:mini
```

You can use any model name; set `OLLAMA_MODEL` in `.env` or rely on the default in `src/config.py`. If `llama3.2:3b` fails with `wrong number of tensors`, the app automatically retries once with `OLLAMA_FALLBACK_MODEL` (default `phi3:mini`). Set `OLLAMA_AUTO_FALLBACK=0` to disable.

## Installation

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` and `torchvision` are listed because **sentence-transformers** depends on **Hugging Face `transformers`**, and Streamlit’s file watcher can trigger optional `transformers` image submodules; those expect `torchvision` to be importable. If you ever see a flood of `ModuleNotFoundError: No module named 'torchvision'`, run `pip install torchvision` (or re-run `pip install -r requirements.txt`).

The first run will download the sentence-transformers embedding model (small, one-time download) into the project’s `.hf_cache/` directory.

## Run the app

1. Ensure Ollama is running and your chosen model is available (`ollama list`).
2. Start Streamlit:

```bash
streamlit run app.py
```

3. Open the URL shown in the terminal (typically `http://localhost:8501`).

## Optional environment variables

Create a `.env` file (not committed) if you want overrides:

```
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_FALLBACK_MODEL=phi3:mini
OLLAMA_AUTO_FALLBACK=1
```

## Run offline evaluation (prompt comparison + figures)

```bash
python scripts/eval_prompts.py
```

Outputs are written to `results/` and `figures/`. See README Evaluation section for metric definitions.

## Logging

See the **Logging policy** section in [README.md](README.md). Logs are written to `logs/app.log` after the first run.

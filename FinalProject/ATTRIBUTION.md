# Attribution

## Data sources

- [`medical_kb.json`](medical_kb.json): **Generated / curated** for this course project. Each entry attributes content style to **MedlinePlus** (U.S. National Library of Medicine) or combined sources as noted in the `source` field. This file is **not** a complete MedlinePlus download.

- [`data/eval_cases.jsonl`](data/eval_cases.jsonl): Synthetic evaluation vignettes authored for this project with **heuristic gold labels** for triage-style categories. Not clinically adjudicated.

## Third-party software

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/) (`langchain-core`, `langchain-community`)
- [NumPy](https://numpy.org/) for cosine-similarity retrieval over precomputed embeddings (project uses `NumpyVectorStore` instead of FAISS)
- [sentence-transformers](https://www.sbert.net/) (embedding model: `all-MiniLM-L6-v2`)
- [Ollama](https://ollama.com/) for local LLM inference
- [scikit-learn](https://scikit-learn.org/) for metrics and confusion matrices
- [matplotlib](https://matplotlib.org/) for figures

## AI-assisted development 

This repository was developed with assistance from Cursor and Claude for debugging and getting a beginning structure for some for the python files to ensure correct skeleton using written out skeleton and pseudo code from me that I fed into the prompt to get a starter code, before continuing to edit through. I also used it to prompt idea generation when missing parts of the project and rubric to ensure that I didn't miss anything from the project overview such as including a logging in process.

Main AI purposes were used to debug and reorganize sturctures of code when errors occurred as well as cleaning up unnecessary code or writings. 

## Model weights

- Embedding weights download automatically on first run (Hugging Face hub).
- LLM weights are managed by **Ollama** locally (`ollama pull <model>`).

# SmartDocsAnalyzer

RAG assistant for analyzing financial documents (PL/EN) with **LlamaIndex + PGVector + FastAPI + Streamlit**.

This is a **junior-friendly portfolio project**: it focuses on a clear end‑to‑end pipeline (upload → parse → index → ask → sources) rather than enterprise features.

## What it does

- Upload a document (PDF / text) and index it into Postgres (pgvector)
- Ask questions in natural language
- Returns an answer **with sources** (filename + page)
- Uses hybrid retrieval + reranking

## Tech stack

- Backend API: FastAPI
- UI: Streamlit
- RAG: LlamaIndex
- Vector store: Postgres + pgvector (via `llama-index-vector-stores-postgres`)
- Optional tracing/evals: LangSmith (tests)

## Architecture (high level)

1. `POST /upload` saves file into `files/` and registers it in DB
2. Parser extracts text (`LlamaParse` if configured)
3. Nodes are created and inserted into PGVector
4. `POST /query` runs the retrieval + rerank + synthesis pipeline
5. Response returns `summary_text` + `source_data`

## Quickstart

### 1) Environment

Create `.env` from the template:

```bash
cp .env.example .env
```

### 2) Start Postgres (pgvector)

```bash
docker compose -f src/docker-compose.yml up -d db
```

### Docker (full stack)

Run DB + API + UI:

```bash
docker compose -f src/docker-compose.yml up --build
```

- API: `http://localhost:8000`
- UI: `http://localhost:8501`

### 3) Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4) Run backend

```bash
PYTHONPATH=. uvicorn src.api.app:app --reload --port 8000
```

### 5) Run UI

```bash
streamlit run src/gui/gui.py
```

## Tests

Unit tests / RAG eval scripts live in `tests/`.

Example:

```bash
PYTHONPATH=. python tests/rag_test/rag_test.py
```

## Notes for GitHub

- This repo intentionally ignores local artifacts (`files/`, `uploads/`, `data/`) and secrets (`.env`).
- Provide your own API keys in `.env`.

## Project quality goals (what this demonstrates)

- Working async API with background jobs
- Database persistence + vector search
- Retrieval pipeline composition (hybrid retrieval + reranker + synthesis)
- Defensive deletion to avoid "zombie" vectors

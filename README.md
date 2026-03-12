# DocuMind — Reasoning-based RAG 

## 1) Overview

DocuMind is a lightweight reasoning-first RAG solution that uses an LLM to
navigate a hierarchical document tree instead of a vector database. Key points:

- Reasoning over retrieval — the model reasons over section summaries and
  decides which branches to explore (human-like navigation).
- No vector DB required — retrieval is performed by the LLM using the
  generated document tree (TOC-like) with page ranges.
- Hierarchical document understanding — documents are converted to a tree of
  sections, each mapped to physical pages and optionally summarized.
- Streaming reasoning — the backend can stream the LLM’s reasoning trace and
  answer chunks to the frontend for a live, auditable experience.

Assumptions
- Only one document is queried at a time. If you do not pass `pdf_path` when
  querying, the API will pick the most recently indexed document in
  `document_store/`.
- Provide your OpenAI-compatible API key in a local `.env` file:

```
OPENAI_API_KEY="sk-..."
```

## 2) How to get the backend started

Prerequisites
- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server

1. Ensure `.env` contains `OPENAI_API_KEY`.
2. Start the FastAPI server:

```bash
uvicorn api:app --reload --port 8000
```

Important folders created at runtime
- `documents/` — uploaded PDF files are saved here.
- `document_store/` — generated JSON indexes are saved here as `<base>_pageindex.json`.

Useful endpoints
- POST /upload — upload a PDF (form-data `file`) and build an index. Returns
  paths and top-level structure.
- POST /query — non-streaming query. Provide `query` (or `question`) and
  optionally `pdf_path`. Returns answer and metadata.
- POST /query/stream — streaming variant (SSE) that emits incremental
  reasoning trace and answer chunks (events: `trace`, `final_trace`,
  `answer_chunk`, `done`).
- GET /status/{doc_id} and GET /status/stream/{doc_id} — document status checks.

Notes
- Indexing uses LLM calls and can be slow for large PDFs. 
- Conversation history is stored in memory for context. Persist if you need long-term
  histories.

## 3) How to get the frontend started

Prerequisites
- Node.js (v16+) and npm / yarn.

Run the frontend

1. Change to the frontend folder:

```bash
cd frontend
```

2. Install dependencies and start dev server:

```bash
npm install
npm run dev    # or `npm start` depending on your setup
```

Configuration
- The frontend expects the backend at `http://localhost:8000` by default. Update
  `API_BASE` in `frontend/src/app/components/*` if your API runs elsewhere.

Usage
- Upload a single PDF via "Manage documents" (this saves the PDF to `documents/`
  and creates an index in `document_store/`).
- Use the chat interface to ask questions. The frontend uses `/query/stream` to
  show live reasoning and stream the final answer.

Troubleshooting
- If `/query` returns HTTP 422, ensure the request body contains JSON with
  the `query` or `question` field and `Content-Type: application/json`.
- If streaming is not working, ensure the browser supports fetch streaming and
  the backend is running via uvicorn.

Credits
- Algorithm and design inspired by VectifyAI/PageIndex (MIT).


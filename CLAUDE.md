# RAG Based AI Teaching Assistant

Fully local, offline-capable intelligent tutoring system for Machine Learning and Deep Learning. Powered by FAISS vector search, Ollama (LLaMA 3.2), Whisper transcription, and a 13-stage data pipeline over 241 YouTube lectures + 2 ML textbooks.

## How to Run

**Prerequisites — must be running before starting the app:**
```
ollama serve                         # Ollama daemon on port 11434
ollama pull llama3.2                 # LLM for answer generation
ollama pull nomic-embed-text         # Embedding model (768-dim)
```

**Start the app (FastAPI — primary entry point):**
```
python main.py
```
Opens at http://127.0.0.1:5000 · API docs at http://127.0.0.1:5000/docs

**Legacy Flask server (kept for reference, not actively maintained):**
```
python app.py
```

**Run evaluation (requires Ollama + FAISS index):**
```
python evaluate.py                   # all 10 test questions
python evaluate.py --questions 3     # quick smoke test
```
Outputs `eval_results.json` and `eval_summary.md`.

## Install Dependencies

```
pip install -r requirements.txt
```

The cross-encoder reranker model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~22 MB) downloads automatically from HuggingFace on first startup.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | **Primary entry point.** FastAPI app — async endpoints, SSE streaming, Pydantic validation |
| `app.py` | Legacy Flask app. Kept intact; do not delete (reference + fallback) |
| `evaluate.py` | Offline RAG evaluation — Faithfulness, Answer Relevancy, Context Precision |
| `templates/index.html` | Chat UI — vanilla JS, SSE streaming consumer, dark/light mode |
| `static/css/style.css` | All styling including streaming cursor animation |
| `BUGFIXES.md` | Documents the 9 critical bugs fixed and why |
| `pipeline/` | 13-stage data ingestion pipeline (run once to build the knowledge base) |

## Generated Files (gitignored, required at runtime)

Both must exist in the project root before starting the app:
- `faiss_with_titles.index` — FAISS `IndexFlatIP` vector index (preferred)
- `faiss.index` — fallback if the above is missing
- `faiss_metadata_clean.json` — parallel array of chunk metadata (text, title, source_url, timestamps)

`main.py` tries `faiss_with_titles.index` first, then falls back to `faiss.index`, then raises a clear `FileNotFoundError` with instructions.

## Architecture

```
Query
  │
  ▼
search_hybrid()          keyword filter on titles + FAISS dot-product
  │                      returns {videos: [...50], books: [...5]}
  │
  ├─► format_sources()   ──► SSE "sources" event (arrives ~200ms after query)
  │
  ▼
search_enhanced()        multi-query: 3 LLM paraphrases, each embedded + searched
  │                      HyDE: hypothetical answer embedded + searched
  │                      all results deduplicated by max score
  │
  ▼
rerank()                 cross-encoder/ms-marco-MiniLM-L-6-v2 reranks top-20 videos
  │
  ▼
_build_context()         top-4 videos (300 chars each) + top-3 books (600 chars each)
  │                      truncated at sentence boundary, max 3000 chars
  ▼
_build_prompt()          system prompt + context assembled
  │
  ▼
Ollama llama3.2          stream=True → tokens yielded via SSE
  │
  ▼
Frontend                 tokens appended to <p>, reformatted into paragraphs on "done"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat UI |
| `POST` | `/ask` | Full answer in one JSON response |
| `POST` | `/ask/stream` | SSE stream: `sources → tokens → done` |
| `GET` | `/docs` | Auto-generated OpenAPI docs (FastAPI) |

**Request body for `/ask` and `/ask/stream`:**
```json
{ "query": "How does backpropagation work?" }
```
Max 500 characters. Validated by Pydantic before any processing.

**SSE event format (`/ask/stream`):**
```
data: {"type": "sources", "sources": {"videos": [...]}}
data: {"type": "token",   "token": "Backpropagation"}
data: {"type": "token",   "token": " is..."}
data: {"type": "done"}
data: {"type": "error",   "message": "..."}   ← only on failure
```

## Data Pipeline (run once to build the knowledge base)

Run these scripts in order from the `pipeline/` directory. Each stage takes the previous output as input.

```
Stage 1:  extract_video_urls.py          → video_urls.json
Stage 2:  json_playlist_creater.py       → video_urls_combined.json
Stage 3:  extract_audios.py              → data/chunks/playlist_ml/, playlist_dl/
Stage 4:  extract_books.py               → data/chunks/books_ml/, books_dl/
Stage 5:  diaganosing_metadata.py        → faiss_metadata_fixed.json
Stage 6:  build_faiss_index.py           → faiss_rebuilt.index
Stage 7:  save_video_embeddings.py       → models/video_embeddings.npy
Stage 8:  merge_metadata.py              → faiss_metadata_complete.json
Stage 9:  diagonositic_text_on_metadata.py  (validation, no output file)
Stage 10: add_book_embeddings.py         → faiss_complete.index
Stage 11: swap_the_index.py              → faiss.index
Stage 12: rebuild_index_with_titles.py   → faiss_with_titles.index  (preferred)
Stage 13: clean_metadata.py              → faiss_metadata_clean.json
```

Stage 3 requires CUDA (GPU). The pipeline supports checkpoint resume — safe to interrupt and re-run.

## Embedding + Index Details

- Model: `nomic-embed-text` via Ollama → 768-dimensional vectors
- Index type: `faiss.IndexFlatIP` (inner product = cosine for L2-normalized vectors)
- All embeddings pre-loaded into `_all_embeddings` numpy array at startup (~seconds, uses RAM)
- Per-query search: `_all_embeddings[candidate_indices] @ q_flat` — one vectorized multiply, no per-chunk reconstruct calls

## Retrieval Pipeline Details

**search_hybrid** (fast, ~200ms):
1. Extract keywords from query, filter stopwords
2. Expand acronyms (CNN→convolution, LSTM→long short, etc.)
3. Filter `video_indices` by keyword match in titles
4. Embed query → batch dot-product with filtered videos + all books
5. Returns top-50 videos + top-5 books by semantic score

**search_enhanced** (runs after sources are sent to client):
1. Call `generate_query_variants()` — LLM generates 3 paraphrases
2. Embed each variant, search all videos + books
3. Call `generate_hypothetical_doc()` — LLM writes a hypothetical textbook passage
4. Embed that passage, search (HyDE technique)
5. `_merge_results()` deduplicates by `(source_url, start)` key, keeps max score

**rerank** (cross-encoder, ~200-400ms on CPU):
- Takes top-20 videos + top-5 books from enhanced pool
- `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (query, passage) pair
- Reranked order is used for `_build_context`; original semantic scores used for `format_sources`
- Graceful fallback if sentence-transformers not installed

## Evaluation

`evaluate.py` implements the three reference-free RAGAS metrics natively via Ollama:

| Metric | How it's measured |
|--------|------------------|
| **Faithfulness** | LLM extracts statements from answer; LLM judges each as supported/unsupported by context |
| **Answer Relevancy** | LLM generates reverse questions from the answer; cosine sim with original question |
| **Context Precision** | LLM judges each retrieved chunk as relevant/irrelevant; MAP-weighted average |

Note: The `ragas` PyPI library (0.4.x) has a broken import against `langchain-community` 0.4.x. The native implementation above is used instead. See `requirements.txt` comment.

## Known Gotchas

- **Startup takes 10-30s** — all embeddings are loaded into RAM. Normal behaviour, not a hang.
- **FAISS ops are blocking C++** — in `main.py` they run inside `asyncio.to_thread()`. Do not call them directly from async context.
- **Reranker score scale ≠ semantic score scale** — cross-encoder logits (~−10 to +10) cannot be compared to cosine similarity scores (0 to 1). `format_sources` always uses the original semantic scores.
- **HyDE adds latency** — 4 extra Ollama calls before LLM streaming starts. This happens *after* the `sources` SSE event, so the UI is not blank during this time.
- **Pipeline typos** — `diaganosing_metadata.py` and `diagonositic_text_on_metadata.py` have typos in their filenames. Do not rename them without updating any scripts that reference them by name.
- **`index.html` uses `/static/css/style.css`** — hardcoded path (not `url_for`). Works with both Flask and FastAPI. Do not revert to `url_for('static', filename=...)` — that is Flask-only syntax.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | FastAPI 0.115+ + uvicorn (async) |
| Vector search | FAISS `IndexFlatIP` (faiss-cpu) |
| LLM | LLaMA 3.2 via Ollama |
| Embeddings | nomic-embed-text (768-dim) via Ollama |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers) |
| Transcription | OpenAI Whisper `base` model on CUDA |
| PDF extraction | pypdf |
| Video download | yt-dlp |
| Async HTTP | httpx (Ollama streaming in FastAPI) |
| Frontend | Vanilla JS, SSE via `fetch` + `ReadableStream` |

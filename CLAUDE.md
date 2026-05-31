# RAG Based AI Teaching Assistant

Production-grade RAG system deployed on HuggingFace Spaces. ML + Deep Learning tutor backed by 241 YouTube lectures + 2 textbooks.

**Live URL:** https://huggingface.co/spaces/ayushthecaringnihilist/rag-ai-teaching

---

## How to Run

**Local (requires Ollama):**
```
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text
python main.py          → http://127.0.0.1:5000
```

**Cloud mode (Groq API, no Ollama):**
```
GROQ_API_KEY=your_key USE_LOCAL_EMBED=true python main.py
```

**Evaluation:**
```
python evaluate.py --questions 5    # ~7 min
python evaluate.py                  # full 10 questions
```

---

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app — primary entry point |
| `app.py` | Legacy Flask — keep, do not delete |
| `evaluate.py` | RAGAS-style evaluation (Faithfulness, Answer Relevancy, Context Precision) |
| `Dockerfile.spaces` | HuggingFace Spaces deployment (rename to Dockerfile when pushing to HF) |
| `requirements-prod.txt` | Cloud deps — no Whisper/yt-dlp/pypdf |
| `render.yaml` | Render.com config (standard plan, 2GB RAM) |
| `deploy_to_hf.ps1` | One-click deploy script to HuggingFace |
| `update_hf.ps1` | Push incremental updates to HuggingFace Space |
| `BUGFIXES.md` | 9 critical bugs documented |
| `INTERVIEW_PREP.md` | Interview Q&A for this project |

---

## Runtime Files (gitignored)

Located in `models/` subdirectory. `main.py` searches project root then `models/`.
- `models/faiss_with_titles.index` — FAISS IndexFlatIP (7,592 vectors, 768-dim)
- `models/faiss_metadata_clean.json` — parallel chunk metadata array

In cloud: downloaded from S3 at startup if `S3_BUCKET` env var is set, OR included in HF Space repo via git-lfs.

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | — | If set, all LLM calls use Groq instead of Ollama |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model to use |
| `USE_LOCAL_EMBED` | `false` | If `true`, use sentence-transformers instead of Ollama for embeddings |
| `S3_BUCKET` | — | If set, downloads FAISS files from S3 at startup |
| `PORT` | `5000` | Server port (Render/HF inject this automatically) |
| `RENDER` | — | If set, binds to `0.0.0.0` instead of `127.0.0.1` |

---

## Architecture

```
User Query
    ↓
search_hybrid()        keyword filter + FAISS dot-product  (~200ms)
    ├──► format_sources() → SSE "sources" event
    ↓
search_enhanced()      multi-query (3 paraphrases) + HyDE (hypothetical doc)
    ↓                  deduplicate by max score
rerank()               cross-encoder/ms-marco-MiniLM-L-6-v2 on top-20
    ↓
_build_context()       top-4 video (300ch) + top-3 book (600ch) chunks, ≤3000ch
    ↓
_stream_llm()          Groq (cloud) OR Ollama (local) — yields tokens
    ↓
Frontend               SSE → markdown rendered on "done" event
```

---

## API Endpoints

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/` | Serves `templates/index.html` as raw HTML |
| `POST` | `/ask/brief` | Fast: no HyDE/multi-query/reranking, `num_predict:120`, ~5-8s |
| `POST` | `/ask/stream` | Full pipeline SSE: `sources → tokens → done` |
| `POST` | `/ask` | Non-streaming full answer |
| `GET` | `/docs` | OpenAPI auto-docs |

Request body: `{"query": "...", "history": [{"role": "user/assistant", "content": "..."}]}`

History capped at last 3 turns (`_MAX_HISTORY_TURNS = 3`). Short follow-ups contextualized via `_contextualize_query()` before FAISS search.

---

## Two-Stage Answer UX

`/ask/brief` → 2-3 sentence definition (~5s) → "Deep Dive →" button → `/ask/stream` (full pipeline, ~30-60s).

Sources always appear in ~200ms from the initial `search_hybrid()` call. The `search_enhanced()` + HyDE runs AFTER sources are sent — no blank UI during the extra LLM calls.

---

## Retrieval Details

`search_hybrid()`: keyword filter on titles (acronym map: CNN→convolution, LSTM→long short) → batch numpy dot-product `_all_embeddings[indices] @ q_flat`.

`search_enhanced()`: generates 3 query variants via `_llm_sync()`, generates 1 HyDE doc via `generate_hypothetical_doc()`, embeds all 4, deduplicates by `(source_url, start)` key.

`rerank()`: cross-encoder logits ≠ cosine similarity scale. `format_sources()` always uses original semantic scores. Reranked order only affects `_build_context()` (LLM context quality).

All embeddings pre-loaded into `_all_embeddings` numpy array at startup — no per-query `index.reconstruct()` calls.

---

## Evaluation Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.773 |
| Answer Relevancy | 0.773 |
| Context Precision | 0.121 |

Context Precision is intentionally low — broad retrieval pool + reranker is the design. Faithfulness and Answer Relevancy are what matter for final answer quality.

---

## Cloud Deployment (HuggingFace Spaces)

- **Platform:** HuggingFace Spaces (free, CPU Basic, 16GB RAM)
- **LLM:** Groq API (`GROQ_API_KEY` secret in Space settings)
- **Embeddings:** sentence-transformers `nomic-ai/nomic-embed-text-v1.5`
- **FAISS files:** uploaded to Space repo via `deploy_to_hf.ps1` (git-lfs)
- **Dockerfile:** `Dockerfile.spaces` (renamed to `Dockerfile` in Space repo)

To redeploy after changes: `powershell -ExecutionPolicy Bypass -File update_hf.ps1`

---

## Known Gotchas

- **numpy<2 required in cloud** — torch 2.2.0+cpu was compiled against NumPy 1.x; NumPy 2.x crashes it.
- **einops required** — `nomic-ai/nomic-embed-text-v1.5` via sentence-transformers needs `einops`.
- **sentence-transformers 2.7.0 pinned** — 3.x/5.x crash on Windows with transformers 5.9.x (tf-keras conflict).
- **FAISS ops are blocking C++** — must call inside `asyncio.to_thread()` in async context.
- **Reranker score ≠ semantic score** — do not mix cross-encoder logits with cosine similarities.
- **HyDE latency is hidden** — runs after `sources` SSE event, so UI is never blank.
- **`index.html` uses hardcoded `/static/css/style.css`** — works with both Flask and FastAPI. Never revert to `url_for()`.
- **Pipeline typos** — `diaganosing_metadata.py` and `diagonositic_text_on_metadata.py` have typos. Do not rename.

---

## Frontend (templates/index.html)

Two-state layout: empty state (hero title centered, input 20vh from bottom) ↔ chat state (glass header, scrollable messages, input at bottom). Transition: 200ms fade+slide via inline style animation (double rAF trick for display change).

Markdown rendered via `renderMarkdown()` — HTML-escaped first (XSS), then bold/italic/code/lists/headings applied. Streaming uses `.textContent`; markdown only applies on `done` event.

Copy button appears on `.bot-bubble` hover. Stop button shows during streaming, cancels `_activeReader.cancel()`.

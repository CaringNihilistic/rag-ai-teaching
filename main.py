"""
RAG AI Teaching Assistant — FastAPI entry point
================================================
Replaces app.py (Flask).  Run with:
    python main.py
    uvicorn main:app --host 127.0.0.1 --port 5000 --reload

Key improvements over Flask version:
  - Async endpoints: FAISS/numpy blocking calls run in a thread pool
  - Native async streaming via StreamingResponse + async generator
  - Pydantic request validation replaces manual .strip() / length checks
  - Auto OpenAPI docs at http://127.0.0.1:5000/docs
"""

import sys
# Windows cp1252 console can't encode emojis — force UTF-8 for all print output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import asyncio
import json
import os
import re
import webbrowser
from collections import defaultdict
from contextlib import asynccontextmanager

import faiss
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Cross-encoder reranker (optional — graceful fallback if not installed)
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _reranker = _CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    RERANKER_AVAILABLE = True
except ImportError:
    _reranker = None
    RERANKER_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
EMBED_MODEL    = "nomic-embed-text"
LLM_MODEL      = "llama3.2"
OLLAMA_EMBED   = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT    = "http://localhost:11434/api/chat"

# ---------------------------------------------------------------------------
# Module-level state (populated in lifespan startup)
# ---------------------------------------------------------------------------
index: faiss.Index = None          # type: ignore[assignment]
metadata: list     = []
video_indices: list = []
book_indices:  list = []
_all_embeddings: np.ndarray = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# LIFESPAN — load heavy resources once, share across all requests
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, metadata, video_indices, book_indices, _all_embeddings

    print("🔄  Loading RAG system …")

    # Resolve index file — check project root then models/ subdirectory
    _search_dirs = [BASE_DIR, os.path.join(BASE_DIR, "models")]
    index_file = None
    for fname in ("faiss_with_titles.index", "faiss.index"):
        for d in _search_dirs:
            candidate = os.path.join(d, fname)
            if os.path.exists(candidate):
                index_file = candidate
                break
        if index_file:
            break
    if not index_file:
        raise FileNotFoundError(
            f"\n❌  No FAISS index found in {BASE_DIR} or {BASE_DIR}/models\n"
            "Run the full pipeline first (stages 1-12)."
        )

    # Resolve metadata file — same search order
    metadata_file = None
    for d in _search_dirs:
        candidate = os.path.join(d, "faiss_metadata_clean.json")
        if os.path.exists(candidate):
            metadata_file = candidate
            break
    if not metadata_file:
        raise FileNotFoundError(f"\n❌  faiss_metadata_clean.json not found in {BASE_DIR} or models/")

    # These are blocking — run in a thread so the event loop stays free
    index    = await asyncio.to_thread(faiss.read_index, index_file)
    raw      = await asyncio.to_thread(open(metadata_file, encoding="utf-8").read)
    metadata = json.loads(raw)

    video_indices = [i for i, c in enumerate(metadata) if c.get("source_type") == "video"]
    book_indices  = [i for i, c in enumerate(metadata) if c.get("source_type") == "book"]

    print("   📦  Pre-loading embeddings into RAM …")
    emb = np.zeros((index.ntotal, index.d), dtype="float32")
    for i in range(index.ntotal):
        emb[i] = index.reconstruct(i)
    _all_embeddings = emb

    reranker_status = "cross-encoder/ms-marco-MiniLM-L-6-v2 (ready)" if RERANKER_AVAILABLE \
                      else "not available — pip install sentence-transformers"
    print(f"   🏆  Reranker: {reranker_status}")
    print(f"✅  System Ready  |  {index.ntotal} vectors  |  "
          f"{len(video_indices)} video chunks  |  {len(book_indices)} book chunks\n")

    # Open browser once (non-blocking)
    asyncio.get_event_loop().call_later(2, lambda: webbrowser.open("http://127.0.0.1:5000"))

    yield
    # Nothing to clean up


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG AI Teaching Assistant",
    description=(
        "Machine Learning & Deep Learning tutor powered by "
        "local RAG (FAISS + Ollama LLaMA 3.2) with cross-encoder reranking."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
_TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "index.html")

# ---------------------------------------------------------------------------
# REQUEST SCHEMA — Pydantic validates + documents automatically
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500,
                       description="Question about ML or Deep Learning (max 500 chars)")

# ---------------------------------------------------------------------------
# HELPERS (synchronous — called inside asyncio.to_thread where needed)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return re.sub(r'\s+', ' ', text).strip()


def embed_query(text: str) -> np.ndarray:
    import requests as _req
    try:
        r = _req.post(OLLAMA_EMBED, json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
        r.raise_for_status()
    except _req.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama — is it running on port 11434?")
    except _req.exceptions.Timeout:
        raise RuntimeError("Ollama embedding timed out.")
    emb = np.array(r.json()["embedding"], dtype="float32")
    faiss.normalize_L2(emb.reshape(1, -1))
    return emb


def extract_keywords(query: str) -> list[str]:
    stops = {
        "what","is","how","do","does","work","works","explain","tell",
        "me","about","the","a","an","are","can","you","please","help",
    }
    return [w.strip("?,!.").lower() for w in query.lower().split()
            if w.strip("?,!.").lower() not in stops]


def search_hybrid(query: str) -> dict:
    keywords = extract_keywords(query)
    acronyms = {
        "cnn":  ["cnn","convolution"], "cnns": ["cnn","convolution"],
        "rnn":  ["rnn","recurrent"],   "rnns": ["rnn","recurrent"],
        "lstm": ["lstm","long short"], "gru":  ["gru","gated"],
        "gan":  ["gan","adversarial"], "vae":  ["vae","variational"],
        "transformer": ["transformer","attention"],
        "backprop": ["backprop","back propagation","chain rule"],
        "gradient": ["gradient","descent"],
    }

    keyword_video_idx = []
    for i in video_indices:
        title = re.sub(r'^\d+[\s_-]*', '', metadata[i].get("title","").lower()).replace(".wav","")
        if any(k in title for k in keywords) or \
           any(k in acronyms and any(a in title for a in acronyms[k]) for k in keywords):
            keyword_video_idx.append(i)

    search_idx = keyword_video_idx if keyword_video_idx else video_indices
    q_flat     = embed_query(query).flatten()

    video_embs   = _all_embeddings[search_idx]
    video_scores = video_embs @ q_flat
    video_results = sorted(
        ({**metadata[i], "score": float(s)} for i, s in zip(search_idx, video_scores)),
        key=lambda x: x["score"], reverse=True
    )

    book_embs   = _all_embeddings[book_indices]
    book_scores = book_embs @ q_flat
    book_results = sorted(
        ({**metadata[i], "score": float(s)} for i, s in zip(book_indices, book_scores)),
        key=lambda x: x["score"], reverse=True
    )

    return {"videos": video_results[:50], "books": book_results[:5]}


# ---------------------------------------------------------------------------
# MULTI-QUERY + HyDE HELPERS
# ---------------------------------------------------------------------------
def _llm_sync(prompt: str, max_tokens: int = 200, temperature: float = 0.5) -> str:
    """Blocking LLM call shared by multi-query and HyDE generators."""
    import requests as _req
    try:
        r = _req.post(
            OLLAMA_CHAT,
            json={
                "model":    LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "options":  {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()
    except Exception:
        return ""


def generate_query_variants(query: str, n: int = 3) -> list[str]:
    """Ask the LLM to rephrase the query n ways. Returns up to n strings."""
    raw = _llm_sync(
        f"Generate {n} different ways to ask the following question about machine learning or "
        f"deep learning. Output only the questions, one per line, no numbering.\n\nQuestion: {query}",
        max_tokens=160,
        temperature=0.7,
    )
    return [l.strip() for l in raw.splitlines() if l.strip() and len(l.strip()) > 10][:n]


def generate_hypothetical_doc(query: str) -> str:
    """
    HyDE — Hypothetical Document Embeddings (Gao et al. 2022).
    Ask the LLM to write a passage that *would* answer the query.
    Embedding that hypothetical passage often retrieves better chunks
    than embedding the question directly.
    """
    return _llm_sync(
        f"Write a short, factual passage (3-4 sentences) from a machine learning textbook "
        f"that directly answers this question:\n\n{query}",
        max_tokens=200,
        temperature=0.3,
    )


def _search_with_embedding(q_flat: np.ndarray) -> dict:
    """Run FAISS dot-product for all video + book indices with the given embedding."""
    vid_scores  = _all_embeddings[video_indices] @ q_flat
    book_scores = _all_embeddings[book_indices]  @ q_flat
    return {
        "videos": sorted(
            ({**metadata[i], "score": float(s)} for i, s in zip(video_indices, vid_scores)),
            key=lambda x: x["score"], reverse=True,
        )[:50],
        "books": sorted(
            ({**metadata[i], "score": float(s)} for i, s in zip(book_indices, book_scores)),
            key=lambda x: x["score"], reverse=True,
        )[:5],
    }


def _merge_results(results_list: list[dict]) -> dict:
    """
    Deduplicate across multiple search result dicts.
    For each unique chunk (keyed by source_url+start or title+start),
    keep whichever appearance had the highest score.
    """
    best_videos: dict = {}
    best_books:  dict = {}
    for r in results_list:
        for c in r.get("videos", []):
            k = (c.get("source_url") or c.get("title",""), c.get("start", 0))
            if k not in best_videos or c["score"] > best_videos[k]["score"]:
                best_videos[k] = c
        for c in r.get("books", []):
            k = (c.get("title",""), c.get("start", 0))
            if k not in best_books or c["score"] > best_books[k]["score"]:
                best_books[k] = c
    return {
        "videos": sorted(best_videos.values(), key=lambda x: x["score"], reverse=True)[:50],
        "books":  sorted(best_books.values(),  key=lambda x: x["score"], reverse=True)[:5],
    }


def search_enhanced(query: str, base_results: dict) -> dict:
    """
    Enhances base_results with multi-query + HyDE retrieval.
    Called AFTER format_sources(base_results) is already streamed to the client,
    so the extra LLM calls don't delay the sources SSE event.

    Multi-query: embed 3 paraphrases, search, merge by max score.
    HyDE: embed a hypothetical answer passage, search, merge by max score.
    """
    all_results = [base_results]

    variants = generate_query_variants(query, n=3)
    print(f"   🔀  Multi-query: {len(variants)} variants generated")
    for v in variants:
        try:
            all_results.append(_search_with_embedding(embed_query(v).flatten()))
        except Exception:
            pass

    hyp_doc = generate_hypothetical_doc(query)
    if hyp_doc:
        print(f"   🧠  HyDE doc: {hyp_doc[:80]}…")
        try:
            all_results.append(_search_with_embedding(embed_query(hyp_doc).flatten()))
        except Exception:
            pass

    merged = _merge_results(all_results)
    print(f"   📊  Enhanced pool: {len(merged['videos'])} videos, {len(merged['books'])} books")
    return merged


def rerank(query: str, results: dict, top_videos: int = 20) -> dict:
    if not RERANKER_AVAILABLE:
        return results
    videos = results.get("videos", [])[:top_videos]
    books  = results.get("books",  [])
    items  = videos + books
    if not items:
        return results
    scores = _reranker.predict([(query, c.get("text","")[:512]) for c in items])
    for c, s in zip(items, scores):
        c["rerank_score"] = float(s)
    return {
        "videos": sorted(videos, key=lambda x: x.get("rerank_score",0), reverse=True),
        "books":  sorted(books,  key=lambda x: x.get("rerank_score",0), reverse=True),
    }


def _build_context(results: dict) -> tuple[str, list, list]:
    blocks, book_texts, video_texts = [], [], []
    for book in results.get("books",[])[:3]:
        text  = clean_text(book.get("text",""))[:600]
        title = re.sub(r'^\d+[\s_-]*','', book.get("title","") or "Reference Book").replace(".pdf","").strip()[:50] or "Reference Book"
        if text:
            book_texts.append(text)
            blocks.append(f"[BOOK] {title}\n{text}")
    for v in results.get("videos",[])[:4]:
        text  = clean_text(v.get("text",""))[:300]
        title = re.sub(r'^\d+[\s_-]*','', v.get("title",""))[:50]
        start = int(v.get("start",0))
        if text:
            video_texts.append(text)
            blocks.append(f"[VIDEO] {title} at {start//60}:{start%60:02d}\n{text}")
    ctx = "\n\n".join(blocks)
    if len(ctx) > 3000:
        trunc = ctx[:3000]
        last  = max(trunc.rfind(". "), trunc.rfind(".\n"))
        ctx   = trunc[:last+1] if last > 1500 else trunc
    return ctx, book_texts, video_texts


def _build_prompt(query: str, context: str) -> str:
    return f"""You are an expert AI tutor for Machine Learning and Deep Learning.

A student asked: "{query}"

Using the sources below, write a thorough and complete explanation:
- Start with a clear definition of the core concept
- Explain WHY it happens and HOW it works in detail
- Give a practical real-world example
- Write at least 4 paragraphs
- Do NOT use markdown symbols like **, ##, or __
- Do NOT include labels like [BOOK] or [VIDEO] in your answer
- Write in plain clear English as if teaching a university student

Sources:
{context}

Answer:"""


def format_sources(results: dict) -> dict:
    groups: dict = defaultdict(list)
    for c in results.get("videos", []):
        if c.get("source_url"):
            groups[c["source_url"]].append(c)
    videos = []
    for url, chunks in sorted(groups.items(),
                               key=lambda x: max(c["score"] for c in x[1]),
                               reverse=True)[:5]:
        best  = max(chunks, key=lambda x: x["score"])
        start = int(best.get("start", 0))
        title = re.sub(r'^\d+[\s_-]*', '', best.get("title",""))
        videos.append({
            "title":     title[:70],
            "url":       f"{url}&t={start}s",
            "timestamp": f"{start//60}:{start%60:02d}",
            "score":     round(best["score"], 3),
        })
    return {"videos": videos}


def generate_answer_sync(query: str, results: dict) -> str:
    import requests as _req
    ctx, book_texts, video_texts = _build_context(results)
    prompt = _build_prompt(query, ctx)
    try:
        r = _req.post(
            OLLAMA_CHAT,
            json={"model": LLM_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False,
                  "options": {"temperature": 0.7, "num_predict": 800}},
            timeout=180,
        )
        if r.status_code == 200:
            answer = r.json().get("message",{}).get("content") or r.json().get("response","")
            if answer.strip():
                return re.sub(r'\*\*|##|__', '', answer).strip()
    except Exception:
        pass
    # Fallback
    parts = []
    for texts, n in ((book_texts, 10), (video_texts[:2], 5)):
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', " ".join(texts)) if len(s.strip())>40]
        if sents:
            parts.append(" ".join(sents[:n]))
    return ("Here is what the course material covers:\n\n" + "\n\n".join(parts)) if parts \
        else "No relevant content found. Try rephrasing."


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def home():
    return HTMLResponse(open(_TEMPLATE_PATH, encoding="utf-8").read())


@app.post("/ask", summary="Get a complete answer (non-streaming)")
async def ask(req: QueryRequest):
    """Returns the full answer + source links in one response."""
    results  = await asyncio.to_thread(search_hybrid, req.query)
    enhanced = await asyncio.to_thread(search_enhanced, req.query, results)  # multi-query + HyDE
    reranked = await asyncio.to_thread(rerank, req.query, enhanced)
    answer   = await asyncio.to_thread(generate_answer_sync, req.query, reranked)
    sources  = format_sources(results)   # original scores for source links
    return {"answer": answer, "sources": sources}


@app.post("/ask/stream", summary="Stream answer tokens via Server-Sent Events")
async def ask_stream(req: QueryRequest):
    """
    Streams the LLM response token-by-token over SSE.
    Event types: sources | token | error | done
    """
    async def generate():
        try:
            # Fast path: original hybrid search → send sources immediately (~200ms)
            results  = await asyncio.to_thread(search_hybrid, req.query)
            sources  = format_sources(results)
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Enhanced path: multi-query + HyDE runs AFTER sources are on screen
            # so the extra LLM calls don't delay the UI response
            enhanced      = await asyncio.to_thread(search_enhanced, req.query, results)
            reranked      = await asyncio.to_thread(rerank, req.query, enhanced)
            context, _, _ = _build_context(reranked)
            prompt        = _build_prompt(req.query, context)

            # Async HTTP to Ollama — doesn't block the event loop
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", OLLAMA_CHAT,
                    json={
                        "model":    LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream":   True,
                        "options":  {"temperature": 0.7, "num_predict": 800},
                    },
                    timeout=httpx.Timeout(180, connect=10),
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                        if chunk.get("done"):
                            break

        except httpx.ConnectError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Cannot connect to Ollama — is it running on port 11434?'})}\n\n"
        except httpx.TimeoutException:
            yield f"data: {json.dumps({'type': 'error', 'message': 'LLM timed out. Try a shorter question.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("🤖  AI TEACHING ASSISTANT  (FastAPI + uvicorn)")
    print("=" * 60)
    print("🌐  http://127.0.0.1:5000")
    print("📖  API docs: http://127.0.0.1:5000/docs")
    print("    Press CTRL+C to quit")
    print("=" * 60 + "\n")

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=5000,
        reload=False,   # set True during dev; causes double-startup
        log_level="info",
    )

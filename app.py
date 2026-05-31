from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import json
import faiss
import numpy as np
import requests
import os
import re
import webbrowser
import threading
from collections import defaultdict

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _reranker = _CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    RERANKER_AVAILABLE = True
except ImportError:
    _reranker = None
    RERANKER_AVAILABLE = False

app = Flask(__name__)

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Bug 1: was hardcoded absolute path

# Try project root then models/ subdirectory for both index and metadata
_search_dirs = [BASE_DIR, os.path.join(BASE_DIR, "models")]

FAISS_INDEX_FILE = None
for _fname in ("faiss_with_titles.index", "faiss.index"):
    for _d in _search_dirs:
        _candidate = os.path.join(_d, _fname)
        if os.path.exists(_candidate):
            FAISS_INDEX_FILE = _candidate
            break
    if FAISS_INDEX_FILE:
        break
if not FAISS_INDEX_FILE:
    FAISS_INDEX_FILE = os.path.join(BASE_DIR, "faiss_with_titles.index")  # triggers clear error below

METADATA_FILE = None
for _d in _search_dirs:
    _candidate = os.path.join(_d, "faiss_metadata_clean.json")
    if os.path.exists(_candidate):
        METADATA_FILE = _candidate
        break
if not METADATA_FILE:
    METADATA_FILE = os.path.join(BASE_DIR, "faiss_metadata_clean.json")  # triggers clear error below

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MAX_QUERY_LEN = 500

# =========================
# LOAD SYSTEM
# =========================
print("🔄 Loading RAG system...")

# Bug 3: crash at startup had no helpful message; now checks files explicitly
if not os.path.exists(FAISS_INDEX_FILE):
    raise FileNotFoundError(
        f"\n❌ FAISS index not found: {FAISS_INDEX_FILE}\n"
        "Run the full pipeline first (stages 1-12).\n"
    )
if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError(
        f"\n❌ Metadata file not found: {METADATA_FILE}\n"
        "Run the full pipeline first (stages 1-12).\n"
    )

index = faiss.read_index(FAISS_INDEX_FILE)

with open(METADATA_FILE, encoding="utf-8") as f:
    metadata = json.load(f)

# Bug 4: was doing a full O(N) metadata scan on every request; pre-index by type at startup
video_indices = [i for i, c in enumerate(metadata) if c.get("source_type") == "video"]
book_indices  = [i for i, c in enumerate(metadata) if c.get("source_type") == "book"]

book_count  = len(book_indices)
video_count = len(video_indices)

# Bug 5: was calling index.reconstruct(idx) in a per-query loop; load all embeddings once
print("   📦 Pre-loading embeddings into RAM...")
_all_embeddings = np.zeros((index.ntotal, index.d), dtype="float32")
for _i in range(index.ntotal):
    _all_embeddings[_i] = index.reconstruct(_i)

if RERANKER_AVAILABLE:
    print("   🏆 Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (ready)")
else:
    print("   ⚠️  Reranker: not available — run: pip install sentence-transformers")

print(f"✅ System Ready!")
print(f"   📊 Index: {index.ntotal} vectors ({FAISS_INDEX_FILE.split(os.sep)[-1]})")
print(f"   📚 Books: {book_count} chunks")
print(f"   🎥 Videos: {video_count} chunks\n")

# =========================
# HELPERS
# =========================
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def embed_query(text):
    try:
        r = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30
        )
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama — is it running on port 11434?")
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama embedding timed out after 30s.")
    emb = np.array(r.json()["embedding"], dtype="float32")
    faiss.normalize_L2(emb.reshape(1, -1))
    return emb

def extract_keywords(query):
    stop_words = {
        'what','is','how','do','does','work','works','explain','tell',
        'me','about','the','a','an','are','can','you','please','help'
    }
    return [
        w.strip('?,!.').lower()
        for w in query.lower().split()
        if w.strip('?,!.').lower() not in stop_words
    ]

# =========================
# HYBRID SEARCH (OPTIMIZED)
# =========================
def search_hybrid(query):
    """Fast hybrid search: filter by keywords first, then semantic search"""
    
    keywords = extract_keywords(query)
    print(f"   🔑 Keywords: {keywords}")
    
    acronyms = {
        "cnn": ["cnn", "convolution"],
        "cnns": ["cnn", "convolution"],
        "rnn": ["rnn", "recurrent"],
        "rnns": ["rnn", "recurrent"],
        "lstm": ["lstm", "long short"],
        "gru": ["gru", "gated"],
        "gan": ["gan", "adversarial"],
        "vae": ["vae", "variational"],
        "transformer": ["transformer", "attention"],
        "backprop": ["backprop", "back propagation", "chain rule"],
        "gradient": ["gradient", "descent"]
    }
    
    # Step 1: Filter videos by title keywords using pre-built index (Bug 4: no O(N) scan)
    keyword_video_indices = []
    for i in video_indices:
        title = metadata[i].get("title", "").lower()
        title_clean = re.sub(r'^\d+[\s_-]*', '', title).replace(".wav", "")

        has_keyword = any(k in title_clean for k in keywords)
        if not has_keyword:
            for k in keywords:
                if k in acronyms and any(a in title_clean for a in acronyms[k]):
                    has_keyword = True
                    break
        if has_keyword:
            keyword_video_indices.append(i)

    # Step 2: Decide which videos to search
    if keyword_video_indices:
        print(f"   ✅ Found {len(keyword_video_indices)} videos with keywords")
        search_video_indices = keyword_video_indices
    else:
        print(f"   📊 No keyword matches, searching all {len(video_indices)} videos")
        search_video_indices = video_indices

    # Step 3: Batch similarity via pre-loaded embeddings (Bug 5: no per-chunk reconstruct loop)
    q_emb = embed_query(query)
    q_flat = q_emb.flatten()

    video_embs  = _all_embeddings[search_video_indices]   # shape (N, d)
    video_scores = video_embs @ q_flat                    # vectorized dot product

    video_results = []
    for rank, idx in enumerate(search_video_indices):
        chunk = metadata[idx].copy()
        chunk["score"] = float(video_scores[rank])
        video_results.append(chunk)
    video_results.sort(key=lambda x: x["score"], reverse=True)

    # Step 4: Book results — same batch approach (Bug 4+5)
    book_embs   = _all_embeddings[book_indices]
    book_scores = book_embs @ q_flat

    book_results = []
    for rank, idx in enumerate(book_indices):
        chunk = metadata[idx].copy()
        chunk["score"] = float(book_scores[rank])
        book_results.append(chunk)
    book_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Show top results
    print(f"   📺 Top 5 videos:")
    for i, v in enumerate(video_results[:5], 1):
        title = re.sub(r'^\d+[\s_-]*', '', v.get("title", ""))[:60]
        print(f"      {i}. {title} ({v['score']:.3f})")
    
    return {
        "videos": video_results[:50],
        "books": book_results[:5]
    }

# =========================
# CONTEXT + PROMPT HELPERS
# Extracted so both the streaming and non-streaming endpoints share the same logic.
# =========================
def _build_context(results):
    """Returns (context_str, book_texts, video_texts). Shared by streaming and non-streaming."""
    context_blocks = []
    book_texts = []
    video_texts = []

    for book in results.get("books", [])[:3]:
        text = clean_text(book.get("text", ""))[:600]
        title = book.get("title", "") or book.get("source", "") or "Reference Book"
        title = re.sub(r'^\d+[\s_-]*', '', title).replace(".pdf", "").strip()[:50] or "Reference Book"
        if text:
            book_texts.append(text)
            context_blocks.append(f"[BOOK] {title}\n{text}")

    for v in results.get("videos", [])[:4]:
        text = clean_text(v.get("text", ""))[:300]
        title = re.sub(r'^\d+[\s_-]*', '', v.get("title", ""))[:50]
        start = int(v.get("start", 0))
        if text:
            video_texts.append(text)
            context_blocks.append(f"[VIDEO] {title} at {start//60}:{start%60:02d}\n{text}")

    context = "\n\n".join(context_blocks)

    if len(context) > 3000:
        truncated = context[:3000]
        last_period = max(truncated.rfind('. '), truncated.rfind('.\n'))
        context = truncated[:last_period + 1] if last_period > 1500 else truncated
        print(f"   ✂️  Context trimmed to {len(context)} chars")

    return context, book_texts, video_texts


def _build_prompt(query, context):
    return f"""You are an expert AI tutor for Machine Learning and Deep Learning.

A student asked: "{query}"

Using the sources below, write a thorough and complete explanation:
- Start with a clear definition of the core concept
- Explain WHY it happens and HOW it works in detail
- Give a practical real-world example
- Explain how to detect or solve it
- Write at least 4 paragraphs
- Do NOT use markdown symbols like **, ##, or __
- Do NOT copy text verbatim, explain naturally in your own words
- Do NOT include labels like [BOOK] or [VIDEO] in your answer
- Write in plain clear English as if teaching a university student

Sources:
{context}

Answer:"""


# =========================
# ANSWER GENERATION (non-streaming, kept as fallback)
# =========================
def generate_answer(query, results):
    context, book_texts, video_texts = _build_context(results)

    print(f"   🧪 Book chunks: {len(book_texts)}, Video chunks: {len(video_texts)}")
    print(f"   🧪 Context: {len(context)} chars")

    prompt = _build_prompt(query, context)
    print(f"   🤖 Sending to LLM (non-streaming)...")

    try:
        r = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 800}
            },
            timeout=180
        )
        if r.status_code == 200:
            answer = r.json().get("message", {}).get("content") or r.json().get("response") or ""
            if answer.strip():
                return re.sub(r'\*\*|##|__', '', answer).strip()
        print(f"   ❌ LLM bad status {r.status_code}")

    except requests.exceptions.Timeout:
        print("   ❌ LLM timed out")
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to Ollama")
    except Exception as e:
        print(f"   ❌ LLM error: {e}")

    # Fallback: return extracted source text
    print("   ⚠️ Using fallback text...")
    parts = []
    for texts, n in ((book_texts, 10), (video_texts[:2], 5)):
        combined = " ".join(texts)
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', combined) if len(s.strip()) > 40]
        if sents:
            parts.append(" ".join(sents[:n]))
    if parts:
        return "Here is what the course material covers:\n\n" + "\n\n".join(parts)
    return "No relevant content found. Please try rephrasing your question."


# =========================
# STREAMING ANSWER (/ask/stream)
# =========================
@app.route("/ask/stream", methods=["POST"])
def ask_stream():
    data = request.json or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "No query provided"}), 400
    if len(query) > MAX_QUERY_LEN:
        return jsonify({"error": f"Query too long (max {MAX_QUERY_LEN} characters)"}), 400

    def generate():
        try:
            # Retrieval is fast (<200ms); send sources before LLM even starts
            results  = search_hybrid(query)
            sources  = format_sources(results)         # original scores for source links
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            reranked           = rerank(query, results)    # cross-encoder rerank
            context, _, _      = _build_context(reranked)  # LLM sees the best-ranked chunks
            prompt             = _build_prompt(query, context)

            resp = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "options": {"temperature": 0.7, "num_predict": 800}
                },
                stream=True,
                timeout=180
            )
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                chunk_data = json.loads(raw_line)
                token = chunk_data.get("message", {}).get("content", "")
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                if chunk_data.get("done"):
                    break

        except requests.exceptions.ConnectionError:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Cannot connect to Ollama — is it running on port 11434?'})}\n\n"
        except requests.exceptions.Timeout:
            yield f"data: {json.dumps({'type': 'error', 'message': 'LLM timed out. Try a shorter question.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # prevents Nginx/proxy from buffering SSE
            "Connection": "keep-alive"
        }
    )

# =========================
# FORMAT SOURCES
# =========================
def format_sources(results):
    videos = []
    groups = defaultdict(list)

    for c in results.get("videos", []):
        url = c.get("source_url")
        if url:
            groups[url].append(c)

    for url, chunks in sorted(
        groups.items(),
        key=lambda x: max(c["score"] for c in x[1]),
        reverse=True
    )[:5]:
        best = max(chunks, key=lambda x: x["score"])
        start = int(best.get("start", 0))
        title = re.sub(r'^\d+[\s_-]*', '', best.get("title", ""))

        videos.append({
            "title": title[:70],
            "url": f"{url}&t={start}s",
            "timestamp": f"{start//60}:{start%60:02d}",
            "score": round(best["score"], 3)
        })

    return {"videos": videos}

# =========================
# RERANKING
# Cross-encoder re-scores the top-N candidates retrieved by hybrid search.
# Only the reranked slice is passed to _build_context (LLM context).
# format_sources still uses the full original results so URL grouping is unaffected.
# =========================
def rerank(query, results, top_videos=20):
    if not RERANKER_AVAILABLE:
        return results

    videos = results.get("videos", [])[:top_videos]
    books  = results.get("books",  [])
    candidates = videos + books
    if not candidates:
        return results

    pairs  = [(query, c.get("text", "")[:512]) for c in candidates]
    scores = _reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    reranked_videos = sorted(videos, key=lambda x: x.get("rerank_score", 0), reverse=True)
    reranked_books  = sorted(books,  key=lambda x: x.get("rerank_score", 0), reverse=True)

    if reranked_videos:
        top = re.sub(r'^\d+[\s_-]*', '', reranked_videos[0].get("title", ""))[:50]
        print(f"   🏆 Reranked — top video: {top} ({reranked_videos[0]['rerank_score']:.2f})")

    return {"videos": reranked_videos, "books": reranked_books}

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Bug 6: no length cap — very long queries could overflow Ollama's context or hang
    if len(query) > MAX_QUERY_LEN:
        return jsonify({"error": f"Query too long (max {MAX_QUERY_LEN} characters)"}), 400
    
    try:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print(f"{'='*60}")
        
        results  = search_hybrid(query)
        reranked = rerank(query, results)          # cross-encoder rerank before LLM context
        answer   = generate_answer(query, reranked)
        sources  = format_sources(results)         # original scores used for source links
        
        print(f"   ✅ Response ready\n")
        
        return jsonify({
            "answer": answer,
            "sources": sources
        })
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =========================
# AUTO BROWSER
# =========================
def open_browser():
    import time
    time.sleep(2)
    try:
        webbrowser.open("http://127.0.0.1:5000")
        print("✅ Browser opened\n")
    except:
        print("⚠️  Please open http://127.0.0.1:5000\n")

if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Thread(target=open_browser, daemon=True).start()
    
    print("="*60)
    print("🤖 AI TEACHING ASSISTANT")
    print("="*60)
    print("🌐 Server: http://127.0.0.1:5000")
    print("   Press CTRL+C to quit")
    print("="*60 + "\n")
    
    # Bug 2: debug=True on 0.0.0.0 exposes the Werkzeug interactive debugger to the whole LAN
    app.run(debug=False, host="127.0.0.1", port=5000)
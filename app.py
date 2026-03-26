from flask import Flask, render_template, request, jsonify
import json
import faiss
import numpy as np
import requests
import os
import re
import webbrowser
import threading
from collections import defaultdict

app = Flask(__name__)

# =========================
# CONFIG
# =========================
BASE_DIR = r"C:\Project- RAG Based Al Teaching"
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "faiss_with_titles.index")  # ← FIXED
METADATA_FILE = os.path.join(BASE_DIR, "faiss_metadata_clean.json")

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# =========================
# LOAD SYSTEM
# =========================
print("🔄 Loading RAG system...")
index = faiss.read_index(FAISS_INDEX_FILE)

with open(METADATA_FILE, encoding="utf-8") as f:
    metadata = json.load(f)

book_count = sum(1 for c in metadata if c.get("source_type") == "book")
video_count = sum(1 for c in metadata if c.get("source_type") == "video")

print(f"✅ System Ready!")
print(f"   📊 Index: {index.ntotal} vectors")
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
    r = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30
    )
    r.raise_for_status()
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
    
    # Step 1: Filter videos by title keywords (FAST - no embedding needed)
    keyword_video_indices = []
    all_video_indices = []
    
    for i, chunk in enumerate(metadata):
        if chunk.get("source_type") == "video":
            all_video_indices.append(i)
            
            title = chunk.get("title", "").lower()
            title_clean = re.sub(r'^\d+[\s_-]*', '', title).replace(".wav", "")
            
            # Check if any keyword in title
            has_keyword = any(k in title_clean for k in keywords)
            
            # Check acronyms
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
        search_indices = keyword_video_indices
    else:
        print(f"   📊 No keyword matches, searching all {len(all_video_indices)} videos")
        search_indices = all_video_indices
    
    # Step 3: Get similarities for filtered videos only (OPTIMIZED)
    q_emb = embed_query(query)
    
    video_results = []
    for idx in search_indices:
        chunk_emb = index.reconstruct(int(idx))
        score = float(np.dot(q_emb.flatten(), chunk_emb))
        
        chunk = metadata[idx].copy()
        chunk["score"] = score
        video_results.append(chunk)
    
    video_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Step 4: Get book results
    book_results = []
    for i, chunk in enumerate(metadata):
        if chunk.get("source_type") == "book":
            chunk_emb = index.reconstruct(int(i))
            score = float(np.dot(q_emb.flatten(), chunk_emb))
            
            chunk = chunk.copy()
            chunk["score"] = score
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
# ANSWER GENERATION
# =========================
def generate_answer(query, results):
    context_blocks = []
    book_texts = []
    video_texts = []

    # Book context
    for book in results.get("books", [])[:3]:
        text = clean_text(book.get("text", ""))[:600]
        
        title = book.get("title", "") or book.get("source", "") or "Reference Book"
        title = re.sub(r'^\d+[\s_-]*', '', title).replace(".pdf", "").strip()[:50]
        title = title if title else "Reference Book"
        
        if text:
            book_texts.append(text)
            context_blocks.append(f"[BOOK] {title}\n{text}")

    # Video context
    for v in results.get("videos", [])[:4]:
        text = clean_text(v.get("text", ""))[:300]
        title = re.sub(r'^\d+[\s_-]*', '', v.get("title", ""))[:50]
        start = int(v.get("start", 0))
        
        if text:
            video_texts.append(text)
            context_blocks.append(f"[VIDEO] {title} at {start//60}:{start%60:02d}\n{text}")

    context = "\n\n".join(context_blocks)

    # Debug prints
    print(f"   🧪 Book chunks: {len(book_texts)}")
    print(f"   🧪 Video chunks: {len(video_texts)}")
    print(f"   🧪 Context length: {len(context)} chars")

    # Keep prompt under 3000 chars to avoid overload
    if len(context) > 3000:
        context = context[:3000]
        print(f"   ✂️  Context trimmed to 3000 chars")

    prompt = f"""You are an expert AI tutor for Machine Learning and Deep Learning.

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

    print(f"   🧪 Prompt length: {len(prompt)} chars")
    print(f"   🤖 Sending to LLM at {OLLAMA_CHAT_URL}...")

    # Try LLM
    try:
        r = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 800
                }
            },
            timeout=180
        )

        print(f"   📡 LLM status: {r.status_code}")

        if r.status_code == 200:
            data = r.json()
            print(f"   🧪 Raw LLM keys: {list(data.keys())}")

            answer = (
                data.get("message", {}).get("content")
                or data.get("response")
                or ""
            )

            print(f"   🧪 Answer length: {len(answer)} chars")
            print(f"   🧪 Answer preview: {answer[:100]}")

            if answer and answer.strip():
                # Strip any leftover markdown symbols
                answer = re.sub(r'\*\*|##|__', '', answer).strip()
                print(f"   ✅ LLM answered successfully!")
                return answer
            else:
                print(f"   ⚠️ LLM returned empty content!")

        else:
            print(f"   ❌ LLM bad status {r.status_code}: {r.text[:300]}")

    except requests.exceptions.Timeout:
        print(f"   ❌ LLM timed out after 180s")
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to Ollama - is it running?")
    except Exception as e:
        print(f"   ❌ LLM error: {type(e).__name__}: {e}")

    # =====================
    # SMART FALLBACK
    # Only triggers if LLM completely fails
    # =====================
    print(f"   ⚠️ LLM failed — using smart fallback...")

    if not book_texts and not video_texts:
        return "No relevant content found. Please try rephrasing your question."

    fallback_parts = []

    if book_texts:
        combined_book = " ".join(book_texts)
        sentences = re.split(r'(?<=[.!?])\s+', combined_book)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 40]
        good_sentences = sentences[:10]
        if good_sentences:
            fallback_parts.append(" ".join(good_sentences))

    if video_texts:
        combined_video = " ".join(video_texts[:2])
        v_sentences = re.split(r'(?<=[.!?])\s+', combined_video)
        v_sentences = [s.strip() for s in v_sentences if len(s.strip()) > 40]
        if v_sentences:
            fallback_parts.append(" ".join(v_sentences[:5]))

    if fallback_parts:
        full_text = "\n\n".join(fallback_parts)
        return (
            f"Here is what the course material covers on this topic:\n\n"
            f"{full_text}\n\n"
            f"For a detailed walkthrough, check the video tutorials below."
        )

    return "Check the video tutorials below for detailed explanations."

# =========================
# FORMAT SOURCES
# =========================
def format_sources(results):
    videos = []
    groups = defaultdict(list)

    for c in results["videos"]:
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
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query", "")
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print(f"{'='*60}")
        
        results = search_hybrid(query)
        answer = generate_answer(query, results)
        sources = format_sources(results)
        
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
    
    app.run(debug=True, host="0.0.0.0", port=5000)
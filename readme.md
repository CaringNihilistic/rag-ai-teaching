# RAG-Based AI Teaching Assistant

> A production-grade RAG system that acts as a personal AI tutor for Machine Learning and Deep Learning.
> Ask any question, get a streamed, reranked, context-grounded explanation with YouTube timestamps.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![Groq](https://img.shields.io/badge/Groq-LLM%20API-purple)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployed-yellow)
![Reranker](https://img.shields.io/badge/Reranker-Cross--Encoder-red)

**Live demo:** https://huggingface.co/spaces/ayushthecaringnihilist/rag-ai-teaching

---

## What It Does

- 🎥 **241 YouTube lectures** fully transcribed via Whisper GPU (Krish Naik ML + DL playlists)
- 📚 **2 authoritative textbooks** — 1,230+ pages chunked and indexed
- ⚡ **Streaming responses** — video sources appear in ~200ms, answer streams token-by-token
- 🏆 **Cross-encoder reranking** — 22MB model re-scores top-20 candidates for accuracy
- 🔀 **Multi-query + HyDE** — generates query paraphrases and hypothetical answers before retrieval
- 🧠 **Multi-turn conversation memory** — follow-up questions reference prior turns
- 📖 **Two-stage UX** — instant 2-sentence definition first, full Deep Dive on demand
- 📊 **RAGAS evaluation** — Faithfulness 0.773 · Answer Relevancy 0.773

---

## Knowledge Base

| Source | Count | Details |
|--------|-------|---------|
| ML Videos | 153 | Krish Naik — Machine Learning playlist |
| DL Videos | 88 | Krish Naik — Deep Learning playlist |
| Book 1 | ~650 pages | *Deep Learning* — Goodfellow, Bengio & Courville |
| Book 2 | ~580 pages | *Hands-On ML with Scikit-Learn, Keras & TensorFlow* — Aurélien Géron |
| **Total** | **241 videos + 1,230 pages** | 7,592 chunks indexed in FAISS |

---

## Architecture

```
User Question
      │
      ▼
search_hybrid()        Keyword filter on video titles + FAISS dot-product (batch numpy)
      │                → {videos: top-50, books: top-5}   ~200ms
      │
      ├──► format_sources() ──► SSE "sources" event   ← user sees sources immediately
      │
      ▼
search_enhanced()      Multi-query: LLM generates 3 paraphrases, each embedded + searched
      │                HyDE: LLM writes hypothetical answer, embed + search
      │                All results deduplicated by max score
      │
      ▼
rerank()               cross-encoder/ms-marco-MiniLM-L-6-v2 re-scores top-20 candidates
      │                (cross-encoder scores query+passage jointly — more accurate than dot-product)
      │
      ▼
_build_context()       Top-4 video chunks (300 chars) + top-3 book chunks (600 chars)
      │                Sentence-boundary truncation at 3,000 chars
      │
      ▼
_stream_llm()          Groq API (cloud) or Ollama (local) — streamed via SSE
      │
      ▼
Browser                Tokens rendered with markdown, reformatted into paragraphs on done
```

---

## Advanced RAG Techniques

**1. Hybrid Search** — keyword filter runs first (zero embedding cost), narrows the candidate pool, then semantic dot-product scores the filtered set. Acronym expansion handles CNN→convolution, LSTM→long short, etc.

**2. Cross-Encoder Reranking** — initial retrieval uses bi-encoder (fast but approximate). Top-20 candidates are re-scored by a cross-encoder that reads query+passage jointly. Bi-encoders compress each independently; cross-encoders see both together — fundamentally more accurate.

**3. HyDE (Hypothetical Document Embeddings)** — based on [Gao et al. 2022](https://arxiv.org/abs/2212.10496). Embed a hypothetical textbook answer instead of the raw question. Documents live in "answer space"; questions live in "question space." Bridging the gap improves retrieval recall.

**4. Multi-Query Retrieval** — LLM generates 3 paraphrases. Each is embedded and searched independently. Results merged by max score. Improves recall when exact phrasing misses relevant chunks.

**5. Two-Stage UX** — `/ask/brief` (fast, no HyDE/reranking, `num_predict:120`) returns a 2-3 sentence definition in ~5-8s. User clicks "Deep Dive →" to trigger `/ask/stream` (full pipeline). Eliminates 45-90s blank wait for simple questions.

**6. Multi-Turn Conversation Memory** — last 3 turns injected into the LLM prompt. Short follow-up queries (< 8 words) are contextualized: "give me an example" + history "What is gradient descent?" → retrieval query becomes "What is gradient descent?. give me an example".

---

## Project Structure

```
rag-ai-teaching/
├── main.py                       ← FastAPI app — primary entry point
├── app.py                        ← Flask app — legacy reference
├── evaluate.py                   ← RAGAS-style evaluation script
├── Dockerfile.spaces             ← HuggingFace Spaces deployment
├── requirements-prod.txt         ← Minimal cloud deps (no Whisper/yt-dlp)
├── render.yaml                   ← Render.com deployment config
├── BUGFIXES.md                   ← 9 critical bugs fixed + explanations
├── INTERVIEW_PREP.md             ← Interview Q&A for this project
├── CLAUDE.md                     ← Project docs for Claude Code
│
├── pipeline/                     ← Run once to build the knowledge base
│   ├── extract_video_urls.py     [Stage 1]
│   ├── json_playlist_creater.py  [Stage 2]
│   ├── extract_audios.py         [Stage 3]  GPU required
│   ├── extract_books.py          [Stage 4]
│   ├── diaganosing_metadata.py   [Stage 5]
│   ├── build_faiss_index.py      [Stage 6]
│   ├── save_video_embeddings.py  [Stage 7]
│   ├── merge_metadata.py         [Stage 8]
│   ├── add_book_embeddings.py    [Stage 10]
│   ├── swap_the_index.py         [Stage 11]
│   └── rebuild_index_with_titles.py [Stage 12]
│
├── templates/index.html          ← Chat UI — SSE streaming, markdown, glassmorphism
├── static/css/style.css          ← Moon background, glass bubbles, animations
└── models/                       ← gitignored — FAISS index + metadata
```

---

## Running Locally

**Prerequisites:** Python 3.10+, Ollama running with `llama3.2` + `nomic-embed-text` pulled

```bash
git clone https://github.com/CaringNihilistic/rag-ai-teaching
cd rag-ai-teaching
pip install -r requirements.txt
python main.py
```

Opens at **http://127.0.0.1:5000** · API docs at **http://127.0.0.1:5000/docs**

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat UI |
| `POST` | `/ask/brief` | Fast path — 2-3 sentence definition, no HyDE/reranking (~5-8s) |
| `POST` | `/ask/stream` | Full pipeline — token-by-token SSE stream |
| `POST` | `/ask` | Non-streaming full answer (JSON) |
| `GET` | `/docs` | OpenAPI docs |

**Request body:**
```json
{ "query": "How does backpropagation work?", "history": [] }
```

**SSE event sequence:**
```
data: {"type": "sources", "sources": {"videos": [...]}}   ← ~200ms
data: {"type": "token",   "token": "Backpropagation"}
data: {"type": "done"}
```

---

## Evaluation Results

| Metric | Score | What It Means |
|--------|-------|---------------|
| **Faithfulness** | **0.773** | 77% of LLM statements are grounded in retrieved context |
| **Answer Relevancy** | **0.773** | Answers semantically address the question asked |
| **Context Precision** | **0.121** | Retrieval pool is broad; reranker selects the best chunks for the LLM |

> Implements [RAGAS paper](https://arxiv.org/abs/2309.15217) metrics natively via Ollama — no external API, no ground truth needed.

```bash
python evaluate.py --questions 5   # quick run (~7 min)
python evaluate.py                 # full 10 questions
```

---

## Deployment

**Live:** https://huggingface.co/spaces/ayushthecaringnihilist/rag-ai-teaching (free, 16GB RAM)

Cloud mode uses **Groq API** for LLM generation (free tier, ~300 tokens/sec) and **sentence-transformers** for embeddings — no Ollama required. Set `GROQ_API_KEY` as a Space secret.

---

## Tech Stack

| Layer | Local | Cloud |
|-------|-------|-------|
| Web framework | FastAPI + uvicorn | same |
| LLM | Ollama llama3.2 | Groq API (llama-3.1-8b-instant) |
| Embeddings | Ollama nomic-embed-text | sentence-transformers nomic-embed-text-v1.5 |
| Vector search | FAISS IndexFlatIP | same |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | same |
| Async HTTP | httpx | same |
| Frontend | Vanilla JS + SSE | same |
| Transcription (pipeline) | OpenAI Whisper base (GPU) | — |
| PDF extraction (pipeline) | pypdf | — |
| Validation | Pydantic v2 | same |

---

## Credits

- **[Krish Naik](https://www.youtube.com/@krishnaik06)** — lecture source material
- **Goodfellow, Bengio, Courville** — *Deep Learning*
- **Aurélien Géron** — *Hands-On ML*
- **RAGAS** — [Shahul Es et al. 2023](https://arxiv.org/abs/2309.15217)
- **HyDE** — [Gao et al. 2022](https://arxiv.org/abs/2212.10496)

# 🤖 RAG-Based AI Teaching Assistant

> A fully local, offline-capable intelligent tutoring system for
> **Machine Learning & Deep Learning** — powered by Whisper, FAISS,
> LLaMA 3.2, and a multi-stage advanced RAG pipeline.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-purple)
![Whisper](https://img.shields.io/badge/Whisper-GPU%20Transcription-yellow)
![Reranker](https://img.shields.io/badge/Reranker-Cross--Encoder-red)

---

## 📌 Overview

A production-grade RAG system that acts as a personal AI tutor for Machine Learning and Deep Learning. Ask any ML/DL question and get a streamed, reranked, context-grounded explanation with YouTube video links timestamped to the exact second.

**Everything runs 100% locally — no OpenAI API, no cloud, no cost per query.**

- 🎥 **241 YouTube lectures** fully transcribed via Whisper GPU
- 📚 **2 authoritative textbooks** — 1,200+ pages chunked and indexed
- ⚡ **Streaming responses** — first token in under a second
- 🏆 **Cross-encoder reranking** — retrieves the most relevant chunks, not just the most similar
- 🔀 **Multi-query + HyDE** — expands the query before retrieval for higher recall
- 📊 **RAGAS-style evaluation** — measurable quality metrics, no ground truth needed

---

## 📊 Knowledge Base

| Source | Count | Details |
|--------|-------|---------|
| 🎥 ML Videos | 153 | Krish Naik — Machine Learning playlist |
| 🎥 DL Videos | 88 | Krish Naik — Deep Learning playlist |
| 📘 Book 1 | ~650 pages | *Deep Learning* — Goodfellow, Bengio & Courville |
| 📘 Book 2 | ~580 pages | *Hands-On ML with Scikit-Learn, Keras & TensorFlow* — Aurélien Géron |
| **Total** | **241 videos + 1,230 pages** | Fully indexed in FAISS |

---

## 🧠 Architecture

```
User Question
      │
      ▼
 search_hybrid()          Keyword filter on video titles
      │                   + FAISS dot-product (batch numpy)
      │                   → {videos: top-50, books: top-5}
      │
      ├──► format_sources() ──► SSE "sources" event  ← arrives ~200ms after query
      │
      ▼
 search_enhanced()        Multi-query: LLM generates 3 paraphrases, each embedded + searched
      │                   HyDE: LLM writes a hypothetical answer passage, embed + search
      │                   All results deduplicated by max score
      │
      ▼
 rerank()                 cross-encoder/ms-marco-MiniLM-L-6-v2
      │                   Re-scores top-20 candidates with a cross-encoder
      │                   (much more accurate than dot-product alone)
      │
      ▼
 _build_context()         Top-4 video chunks (300 chars) + top-3 book chunks (600 chars)
      │                   Sentence-boundary truncation at 3,000 chars
      │
      ▼
 Ollama llama3.2          stream=True → tokens yielded one by one
      │
      ▼
 Browser                  Tokens streamed into live <p> tag, reformatted on completion
```

---

## ⚡ Advanced RAG Techniques

### 1. Streaming Responses
Answers stream token-by-token via **Server-Sent Events**. Sources appear within ~200ms (just FAISS lookup time) while the LLM is still generating. A blinking cursor shows the model is "typing."

### 2. Cross-Encoder Reranking
After the initial dense retrieval (dot-product similarity), the top-20 candidates are passed through `cross-encoder/ms-marco-MiniLM-L-6-v2` — a 22MB model that scores each `(query, passage)` pair jointly. This is far more accurate than embedding similarity alone and typically improves answer quality significantly.

### 3. Multi-Query Retrieval
The LLM generates 3 different phrasings of the user's question. Each phrasing is embedded and searched independently. Results are merged by keeping the highest score for each unique chunk. This improves recall for questions where the phrasing matters for keyword matching.

### 4. HyDE (Hypothetical Document Embeddings)
Based on [Gao et al. 2022](https://arxiv.org/abs/2212.10496). Instead of embedding the question, the LLM first writes a hypothetical textbook passage that *would* answer it. That passage is embedded and used for retrieval. Since document embeddings live in "answer space" rather than "question space," this often retrieves more relevant chunks.

### 5. Hybrid Search
Combines keyword filtering (matching query terms against video titles, with acronym expansion: CNN→convolution, LSTM→long short, etc.) with semantic vector search. The keyword filter runs first with zero embedding cost, narrowing the candidate pool before semantic scoring.

---

## 📁 Project Structure

```
rag-ai-teaching/
│
├── main.py                       ← FastAPI app — primary entry point
├── app.py                        ← Flask app — legacy, kept for reference
├── evaluate.py                   ← RAG quality evaluation script
├── requirements.txt
├── CLAUDE.md                     ← Project docs for Claude Code
├── BUGFIXES.md                   ← 9 critical bugs fixed (documented)
│
├── pipeline/                     ← Run once to build the knowledge base
│   ├── extract_video_urls.py     [Stage 1]  Pull YouTube playlist URLs
│   ├── json_playlist_creater.py  [Stage 2]  Standardise playlist JSON
│   ├── extract_audios.py         [Stage 3]  Whisper GPU transcription
│   ├── extract_books.py          [Stage 4]  Chunk PDF textbooks
│   ├── diaganosing_metadata.py   [Stage 5]  Fuzzy-match video URLs
│   ├── build_faiss_index.py      [Stage 6]  Generate embeddings + index
│   ├── save_video_embeddings.py  [Stage 7]  Cache video embeddings
│   ├── merge_metadata.py         [Stage 8]  Merge video + book metadata
│   ├── diagonositic_text_on_metadata.py [Stage 9] Validate metadata
│   ├── add_book_embeddings.py    [Stage 10] Add book vectors to index
│   ├── swap_the_index.py         [Stage 11] Activate index
│   └── rebuild_index_with_titles.py [Stage 12] Rebuild with title search
│
├── templates/
│   └── index.html                ← Chat UI (streaming SSE consumer)
│
├── static/css/
│   └── style.css                 ← Styling + dark mode + streaming cursor
│
├── data/                         ← ⚠️ gitignored — regenerate via pipeline
│   ├── audios/                   ← ML lecture audio files
│   ├── audios_1/                 ← DL lecture audio files
│   ├── Books/                    ← PDF textbooks
│   └── chunks/                   ← Generated JSON transcript chunks
│
└── [project root — gitignored runtime files]
    ├── faiss_with_titles.index   ← Primary FAISS index (Stage 12 output)
    ├── faiss.index               ← Fallback index (Stage 11 output)
    └── faiss_metadata_clean.json ← Chunk metadata array
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (for Whisper transcription in the pipeline)
- [Ollama](https://ollama.com) installed

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-ai-teaching.git
cd rag-ai-teaching
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

The cross-encoder reranker model (~22 MB) downloads from HuggingFace automatically on first startup.

### 3. Pull Ollama models
```bash
ollama pull nomic-embed-text   # embedding model
ollama pull llama3.2           # LLM for answer generation
```

### 4. Add source data
```
data/audios/     ← ML lecture audio files (.wav / .mp3)
data/audios_1/   ← DL lecture audio files
data/Books/      ← PDF textbooks
```

---

## ⚙️ Running the Pipeline

> Run once to build the full knowledge base. Whisper transcription takes several hours on first run; the pipeline supports checkpoint resume — safe to interrupt at any stage.

```bash
python pipeline/extract_video_urls.py
python pipeline/json_playlist_creater.py
python pipeline/extract_audios.py              # GPU required
python pipeline/extract_books.py
python pipeline/diaganosing_metadata.py
python pipeline/build_faiss_index.py
python pipeline/save_video_embeddings.py
python pipeline/merge_metadata.py
python pipeline/diagonositic_text_on_metadata.py
python pipeline/add_book_embeddings.py
python pipeline/swap_the_index.py
python pipeline/rebuild_index_with_titles.py   # creates faiss_with_titles.index
```

---

## ▶️ Launch the App

Make sure Ollama is running, then:

```bash
python main.py
```

Opens automatically at **http://127.0.0.1:5000** · API docs at **http://127.0.0.1:5000/docs**

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat UI |
| `POST` | `/ask` | Full answer in a single JSON response |
| `POST` | `/ask/stream` | Token-by-token SSE stream |
| `GET` | `/docs` | Interactive OpenAPI documentation |

**Request body:**
```json
{ "query": "How does backpropagation work?" }
```

**SSE stream event sequence (`/ask/stream`):**
```
data: {"type": "sources", "sources": {"videos": [...]}}   ← ~200ms
data: {"type": "token",   "token": "Backpropagation"}
data: {"type": "token",   "token": " is the algorithm..."}
...
data: {"type": "done"}
```

---

## 📊 Evaluation

Run the offline quality evaluation against 10 ML/DL test questions:

```bash
python evaluate.py              # full run
python evaluate.py --questions 3   # quick smoke test
```

Outputs `eval_results.json` and `eval_summary.md`.

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Fraction of answer statements grounded in retrieved context |
| **Answer Relevancy** | Cosine similarity between question and reverse-generated questions from the answer |
| **Context Precision** | Fraction of retrieved chunks judged relevant by the LLM |

> Metrics follow the [RAGAS paper](https://arxiv.org/abs/2309.15217) definitions, implemented with local Ollama — no external API needed.

**Results** *(5 questions, llama3.2, nomic-embed-text, RTX 3050)*:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Faithfulness | **0.773** | 77% of answer statements grounded in retrieved context |
| Answer Relevancy | **0.773** | Answers address the question with high semantic alignment |
| Context Precision | **0.121** | Low — retrieved pool is broad; top chunks are relevant but many are noisy |

> Context Precision is low because the retrieval pool (50 videos + 5 books) is large and many
> chunks are tangentially related. The reranker mitigates this by ensuring the LLM only sees
> the top-ranked chunks — Faithfulness and Answer Relevancy reflect that final quality.
> Run `python evaluate.py` to regenerate with your own models.

---

## 💡 Example Questions

```
"How does backpropagation work?"
"What is the vanishing gradient problem and how do LSTMs solve it?"
"Explain CNNs with a real-world example"
"What is the difference between bagging and boosting?"
"How does the attention mechanism in transformers work?"
"What is overfitting and how do I fix it?"
"Explain batch normalisation and why it helps training"
"What is the bias-variance tradeoff?"
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | FastAPI 0.115+ + uvicorn (async) |
| Async HTTP | httpx (Ollama streaming) |
| Vector search | FAISS `IndexFlatIP` (faiss-cpu) |
| Embeddings | Ollama `nomic-embed-text` (768-dim) |
| LLM | Ollama `llama3.2` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers) |
| Transcription | OpenAI Whisper `base` — GPU (CUDA) |
| PDF extraction | pypdf |
| Video download | yt-dlp |
| Frontend | Vanilla JS, SSE via `fetch` + `ReadableStream` |
| Validation | Pydantic v2 |

---

## ⚠️ Notes

- `data/` and the index files are **gitignored** — run the pipeline to regenerate after cloning
- Whisper requires a **CUDA-capable GPU** (tested on RTX 3050 4GB)
- Pipeline saves **checkpoints** automatically — safe to interrupt and resume
- App **startup takes 10–30s** — all embeddings are pre-loaded into RAM (normal, not a hang)
- **HyDE + multi-query add latency** to the enhanced retrieval pass, but this happens *after* sources are already on screen, so the UI is never blank

---

## 🙏 Credits

- **[Krish Naik](https://www.youtube.com/@krishnaik06)** — ML & DL lecture playlists used as the primary video source
- **Ian Goodfellow, Yoshua Bengio, Aaron Courville** — *Deep Learning*
- **Aurélien Géron** — *Hands-On ML with Scikit-Learn, Keras & TensorFlow*
- **RAGAS** — [Shahul Es et al. 2023](https://arxiv.org/abs/2309.15217) — evaluation metric definitions
- **HyDE** — [Gao et al. 2022](https://arxiv.org/abs/2212.10496) — hypothetical document embeddings

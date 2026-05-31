# Interview Preparation — RAG AI Teaching Assistant

> Written from the perspective of a senior SWE interviewer.
> If you cannot answer a question confidently and concisely, you don't truly understand it yet.
> Every answer here should take you 60-90 seconds to say out loud — not 10 seconds, not 5 minutes.

---

## Things You Must Know By Heart (no hesitation allowed)

**Numbers:**
- 7,592 total chunks in the FAISS index
- 768-dimensional embeddings (nomic-embed-text)
- 241 videos + 1,230 pages = the knowledge base
- RAGAS scores: Faithfulness 0.773 · Answer Relevancy 0.773 · Context Precision 0.121
- Sources appear in ~200ms. Brief answer in ~5-8s. Full Deep Dive ~30-60s.
- Top-50 videos + top-5 books from initial retrieval
- Reranker re-scores top-20 candidates
- 3 query paraphrases generated for multi-query
- Last 3 turns kept in conversation history

**Papers you must cite:**
- HyDE: Gao et al. 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels"
- RAGAS: Shahul Es et al. 2023 — "RAGAS: Automated Evaluation of Retrieval Augmented Generation"

**The retrieval pipeline order (say it without looking):**
`search_hybrid → format_sources (SSE) → search_enhanced → rerank → _build_context → _build_prompt → _stream_llm`

---

## Round 1: Baseline Understanding

**Q: What is RAG and why did you build it instead of just using an LLM directly?**

LLMs have a knowledge cutoff and hallucinate when asked about specific content they weren't trained on. RAG — Retrieval Augmented Generation — grounds the LLM's answer in actual retrieved documents. Instead of relying on parametric memory, you give the model the relevant text at inference time. In my case, the LLM wouldn't know the exact timestamps of a Krish Naik lecture or the specific wording from Goodfellow's textbook. RAG makes the answers verifiably grounded and citable.

---

**Q: Explain your architecture in 2 minutes.**

A user query hits two paths simultaneously conceptually — a fast retrieval path and a deep enhancement path. The fast path does keyword filtering on video titles (sub-millisecond) followed by a vectorized FAISS dot-product search across 7,592 pre-loaded embeddings. This returns video sources to the user in about 200ms via Server-Sent Events. Simultaneously, the enhanced path generates 3 paraphrases of the query and a hypothetical textbook answer using the LLM. All four embeddings are searched, results are deduplicated and merged. The top-20 candidates go through a cross-encoder reranker which scores each query-passage pair jointly. The top-ranked chunks form the LLM context, and the answer streams back token-by-token via Groq API in the cloud or Ollama locally.

---

**Q: What is a vector embedding? How does similarity search work in FAISS?**

An embedding model maps text to a fixed-size dense vector in a high-dimensional space where semantically similar text lands geometrically close. I use nomic-embed-text which outputs 768-dimensional vectors. FAISS (Facebook AI Similarity Search) stores these vectors and enables efficient nearest-neighbor lookup. I use IndexFlatIP — inner product search. After L2-normalizing all vectors, inner product equals cosine similarity. At query time I embed the query, normalize it, and compute a dot-product with all stored vectors using a single numpy matrix multiply: `_all_embeddings[indices] @ q_flat`. No per-chunk reconstruct loop — all embeddings are pre-loaded into RAM at startup, which is why a key bug fix was pre-loading them once rather than calling `index.reconstruct()` 7,592 times per query.

---

## Round 2: Technical Depth

**Q: What's the difference between a bi-encoder and a cross-encoder? Why use both?**

A bi-encoder encodes the query and each document independently into separate vectors, then measures similarity with dot-product. It's fast — you precompute document vectors — but it loses fine-grained interaction between query and document. A cross-encoder takes both the query and the document concatenated as input, running them through the same transformer together. This allows full attention between every token in the query and every token in the passage. It's much more accurate but can't precompute — you must run inference for every (query, document) pair at search time. My system uses bi-encoder for retrieval (fast, scales to thousands of chunks) and cross-encoder for reranking the top-20 results only (accurate but limited by the number of pairs, not the full index).

---

**Q: Explain HyDE. Why does it help? When would it fail?**

HyDE stands for Hypothetical Document Embeddings, from Gao et al. 2022. The core insight: questions and answers live in different embedding spaces. "What is backpropagation?" embeds differently from a textbook passage that answers it. HyDE bridges this gap — instead of embedding the question, you prompt the LLM to generate a hypothetical textbook passage that would answer the question, then embed that passage. Since it's in "answer space," it aligns better with the actual stored documents. It helps for short, ambiguous queries where the question words have low overlap with technical passages. It fails when the LLM hallucinates — generates a confident wrong answer as the hypothesis. My system mitigates this by running HyDE alongside the original query and taking the union, so a bad hypothesis doesn't destroy recall.

---

**Q: Why is Context Precision only 0.121 and does that mean the system is bad?**

No — Context Precision measures whether the entire retrieval pool (50 videos + 5 books) is relevant. My retrieval intentionally casts a wide net. The number looks low because 45 of 50 videos might be tangentially related. What matters is that the reranker correctly selects the 4 most relevant chunks for the LLM context — and Faithfulness at 0.773 shows those 4 chunks are being used well. Context Precision and Faithfulness are measuring different stages of the pipeline. I designed for high recall (broad retrieval) + high precision at the context-building stage (tight reranking), not precision across the full pool.

---

**Q: How does your SSE streaming work technically?**

Server-Sent Events is a one-directional HTTP connection where the server pushes data to the client. In FastAPI I return a `StreamingResponse` with `media_type="text/event-stream"`. Each yielded string must follow the format `data: {json}\n\n` — the double newline is the SSE delimiter. The frontend uses `fetch()` with `response.body.getReader()` to get a `ReadableStream`, reads raw bytes, decodes them, and splits on newline to parse individual SSE events. The tricky part: a single `read()` call can return bytes spanning multiple events or a partial event, so I maintain a buffer and only process complete lines. My system sends four event types: `sources` (arrives ~200ms), `token` (one per LLM token), `error`, and `done`. The `sources` event fires before the LLM even starts generating — the retrieval is fast enough that sources appear while the LLM is still being prompted.

---

**Q: Why did you migrate from Flask to FastAPI? What's the core technical difference?**

Flask is WSGI — synchronous, one request per thread. FastAPI is ASGI — async, one event loop handles many concurrent requests via coroutines. In a RAG system this matters because FAISS operations are blocking C++ code and Ollama API calls take 2-30 seconds. In Flask, blocking one request blocks the thread for that user. In FastAPI, I use `asyncio.to_thread()` to push blocking operations into a thread pool while the event loop stays free for other requests. Additionally, FastAPI auto-generates OpenAPI docs from Pydantic schemas, validates request bodies at the framework level before my code runs, and has native `StreamingResponse` — no `stream_with_context` hack. The migration also let me write a cleaner `_stream_llm()` async generator that's a single source of truth for both streaming endpoints.

---

## Round 3: Production Engineering

**Q: Walk me through the 9 critical bugs you fixed.**

The most impactful ones: A hardcoded absolute Windows path `r"C:\Project..."` made the app non-portable — fixed with `os.path.dirname(os.path.abspath(__file__))`. Running Flask with `debug=True` and `host="0.0.0.0"` exposed the Werkzeug interactive debugger to the entire LAN — anyone could execute arbitrary Python. Two O(N) performance bugs: a full metadata scan on every query (fixed with pre-built `video_indices`/`book_indices` lists at startup) and per-chunk `index.reconstruct()` calls in a loop (fixed by pre-loading all embeddings into a numpy array, reducing per-query reconstruction to a single matrix multiply). Missing startup guard meant cryptic crash with no helpful message if FAISS files were missing. A wrong package (`difflib2` instead of stdlib `difflib`) that could silently corrupt fuzzy matching. Context was truncated mid-word instead of at sentence boundaries. No query length validation allowed prompt injection or Ollama OOM. FAISS index filename mismatch between what the pipeline output and what the app loaded.

---

**Q: How does your conversation memory work and what are its limitations?**

The last 3 turns are stored client-side as a JavaScript array of `{role, content}` objects. On every request the frontend sends this history alongside the query. On the server, `_build_prompt()` injects history as a "Conversation so far: Student: … / Tutor: …" block. Short follow-up queries (< 8 words) are contextualized before FAISS search via `_contextualize_query()` — "give me an example" + history "What is gradient descent?" becomes "What is gradient descent?. give me an example" for retrieval. Limitations: history resets on page refresh (client-side storage), the 3-turn cap means longer conversations lose early context, retrieval for follow-ups still isn't perfect because we only prepend the last user question to the search query rather than a proper query rewriting step.

---

**Q: You mentioned a security vulnerability. What was it exactly and what's the attack surface?**

Running `app.run(debug=True, host="0.0.0.0", port=5000)` in Flask activates the Werkzeug debugger and binds to all network interfaces. The Werkzeug debugger provides an interactive Python REPL in the browser — anyone who can reach port 5000 (anyone on the same Wi-Fi network, LAN, or if port 5000 was accidentally exposed to the internet) can execute arbitrary Python code with the process's permissions. That means reading files, spawning processes, exfiltrating data, or using the machine as a pivot. Fixed by setting `debug=False` and `host="127.0.0.1"`.

---

## Round 4: System Design

**Q: How would you scale this to 10,000 concurrent users?**

As-is, the bottleneck is the LLM inference (Groq handles this) and the FAISS search (currently a numpy matrix multiply in a thread pool — fast but CPU-bound). For 10k users I'd: (1) Switch to Qdrant or Weaviate as the vector store — they support sharding, filtering, and concurrent access natively. (2) Use a proper async embedding service or OpenAI's embedding API to handle concurrent embedding requests. (3) Add a Redis cache for queries that have been seen before — the same question asked many times should hit the cache, not run the full pipeline. (4) Deploy behind a load balancer with multiple FastAPI instances. (5) The reranker becomes a bottleneck — could batche cross-encoder inference or use a lighter model.

---

**Q: What would you improve about the retrieval quality?**

The biggest remaining gap is that the FAISS index was built with Ollama's quantized embedding model, but in cloud mode I use sentence-transformers with full-precision weights. These are slightly different embedding spaces — the nearest neighbors are mostly the same but not identical. The right fix is contextual retrieval: before embedding each chunk, prepend a 1-2 sentence context summary generated by an LLM ("This is from Chapter 5 of Deep Learning, discussing backpropagation"). Anthropic found this reduces retrieval failures by ~49%. The other improvement would be a proper query rewriting step for conversation follow-ups rather than the simple string concatenation I do now.

---

## Round 5: ML/RAG Concepts to Know Cold

**Q: Why does L2 normalization make inner product equivalent to cosine similarity?**

Cosine similarity between vectors A and B is `dot(A,B) / (||A|| * ||B||)`. If both vectors are already L2-normalized (i.e., `||A|| = ||B|| = 1`), the denominator is 1·1 = 1, so cosine similarity reduces to just `dot(A,B)` — which is the inner product. FAISS's `IndexFlatIP` computes inner products. By normalizing all vectors to unit length before insertion and normalizing queries before search, I get exact cosine similarity from the inner product index. This is why `faiss.normalize_L2()` is called on every embedding.

---

**Q: What is chunking and why does overlap matter?**

Large documents can't be embedded as a whole — embedding models have token limits (~8,192 for nomic-embed-text) and longer texts produce averaged embeddings that lose specificity. Chunking splits documents into smaller segments. Overlap between consecutive chunks (e.g., 30 words overlap on 180-word chunks) ensures that a sentence near the boundary of a chunk isn't missed — without overlap, the split might land mid-concept and neither chunk contains the full idea. Too much overlap wastes storage and retrieval bandwidth. Too little misses boundary content. My books use 180 words per chunk with 30-word overlap; videos use ~120-word chunks from the Whisper timestamped segments.

---

**Q: What is the difference between Faithfulness and Answer Relevancy in RAGAS?**

Faithfulness asks: "Is everything the LLM said actually supported by the retrieved context?" It's a precision metric — it punishes hallucination. I measure it by extracting individual statements from the answer and asking the LLM to judge each one against the context. Answer Relevancy asks: "Does the answer address the question that was asked?" It's measured by having the LLM generate reverse questions from the answer and computing cosine similarity between those reverse questions and the original query. A high-faithfulness, low-relevancy answer would be one that only quotes from the context but doesn't actually answer what was asked. A high-relevancy, low-faithfulness answer would correctly answer the question but with statements not in the retrieved documents — hallucination.

---

## Questions You Will Definitely Get — Prepare 60-second Answers

1. "Tell me about a technical challenge you faced in this project and how you solved it."
   → Use the O(N) metadata scan bug + the numpy matrix multiply fix. Measurable impact.

2. "Why did you choose this tech stack over alternatives like LangChain or LlamaIndex?"
   → You built the RAG components from scratch to understand them deeply. LangChain adds abstraction overhead and hides what's happening. Knowing the FAISS API, the SSE format, and the chunking logic directly lets you debug and optimize in ways a framework user cannot.

3. "How did you measure the quality of your system?"
   → RAGAS metrics. Explain the three metrics and your specific scores. Acknowledge that Context Precision being low is a design choice, not a flaw.

4. "What would you do differently if you started over?"
   → Start with FastAPI (not Flask). Implement contextual retrieval from the beginning. Use a proper vector database (Qdrant) instead of raw FAISS for production — better filtering, persistence, concurrent access.

5. "What is the live demo URL?"
   → https://huggingface.co/spaces/ayushthecaringnihilist/rag-ai-teaching

---

## Red Flags That Will Kill Your Interview

- Saying "I just used LangChain" or "I just called the API" — you built this from scratch, own it.
- Not knowing what 768 dimensions means or why you L2-normalize.
- Confusing the bi-encoder (retrieval) with the cross-encoder (reranking).
- Not being able to explain why Context Precision is 0.121 without panicking.
- Not knowing the HyDE paper or treating it as a vague "trick."
- Saying "the app runs locally" without mentioning the HuggingFace deployment.
- Not being able to draw the architecture from memory on a whiteboard.

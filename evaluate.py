"""
RAG Evaluation Script
=====================
Measures three reference-free RAGAS metrics using the project's own Ollama stack.
No external API keys or ground-truth answers required.

Metrics implemented (same definitions as ragas paper):
  - Faithfulness        : fraction of answer statements grounded in retrieved context
  - Answer Relevancy    : cosine similarity between question and answer embeddings
  - Context Precision   : fraction of retrieved chunks that are actually relevant

Run:
    python evaluate.py
    python evaluate.py --questions 5   # quick smoke-test

Output:
    eval_results.json      raw per-question data
    eval_summary.md        human-readable summary for README/portfolio
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Bootstrap: reuse the same FAISS + metadata the app uses
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

# Load FAISS index
for _fname in ("faiss_with_titles.index", "faiss.index"):
    _path = BASE_DIR / _fname
    if _path.exists():
        INDEX_FILE = str(_path)
        break
else:
    sys.exit("❌  No FAISS index found. Run the full pipeline first.")

METADATA_FILE = BASE_DIR / "faiss_metadata_clean.json"
if not METADATA_FILE.exists():
    sys.exit(f"❌  Metadata not found: {METADATA_FILE}")

print("🔄  Loading index + metadata …")
index    = faiss.read_index(INDEX_FILE)
metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

video_indices = [i for i, c in enumerate(metadata) if c.get("source_type") == "video"]
book_indices  = [i for i, c in enumerate(metadata) if c.get("source_type") == "book"]

_all_emb = np.zeros((index.ntotal, index.d), dtype="float32")
for _i in range(index.ntotal):
    _all_emb[_i] = index.reconstruct(_i)

print(f"✅  Index ready — {index.ntotal} vectors\n")

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------
EMBED_MODEL     = "nomic-embed-text"
LLM_MODEL       = "llama3.2"
OLLAMA_EMBED    = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT     = "http://localhost:11434/api/chat"

def _embed(text: str) -> np.ndarray:
    r = requests.post(OLLAMA_EMBED, json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    r.raise_for_status()
    v = np.array(r.json()["embedding"], dtype="float32")
    faiss.normalize_L2(v.reshape(1, -1))
    return v

def _chat(prompt: str, max_tokens: int = 300) -> str:
    r = requests.post(
        OLLAMA_CHAT,
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_tokens},
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "").strip()

# ---------------------------------------------------------------------------
# Retrieval (mirrors app.py search_hybrid, without keyword filtering for simplicity)
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k_videos: int = 10, top_k_books: int = 3):
    q = _embed(query).flatten()

    vid_scores = _all_emb[video_indices] @ q
    vid_top    = sorted(zip(video_indices, vid_scores), key=lambda x: -x[1])[:top_k_videos]

    book_scores = _all_emb[book_indices] @ q
    book_top    = sorted(zip(book_indices, book_scores), key=lambda x: -x[1])[:top_k_books]

    contexts = [metadata[i]["text"] for i, _ in vid_top + book_top if metadata[i].get("text")]
    return contexts

# ---------------------------------------------------------------------------
# Generate answer (non-streaming, same prompt as app.py)
# ---------------------------------------------------------------------------
_PROMPT = """You are an expert AI tutor for Machine Learning and Deep Learning.

A student asked: "{query}"

Using the sources below, write a concise and accurate explanation.
Do NOT use markdown. Do NOT include [BOOK] or [VIDEO] labels.
Write 2-3 paragraphs in plain English.

Sources:
{context}

Answer:"""

def generate(query: str, contexts: list[str]) -> str:
    ctx = "\n\n".join(contexts)[:2500]
    return _chat(_PROMPT.format(query=query, context=ctx), max_tokens=600)

# ---------------------------------------------------------------------------
# METRIC 1 — Faithfulness
# Definition (RAGAS paper): fraction of answer statements that are
# entailed by the retrieved context.  Uses LLM-as-judge.
# ---------------------------------------------------------------------------
_FAITH_EXTRACT = """List every factual claim in the following answer as short, numbered statements.
Only list statements, one per line, no commentary.

Answer:
{answer}"""

_FAITH_JUDGE = """For each statement below, reply ONLY with "yes" or "no" on its own line
(same order, one line per statement).
"yes" means the statement is fully supported by the context.
"no" means it is not supported or contradicts the context.

Context:
{context}

Statements:
{statements}"""

def faithfulness(answer: str, contexts: list[str]) -> float:
    raw = _chat(_FAITH_EXTRACT.format(answer=answer), max_tokens=300)
    statements = [l.lstrip("0123456789.) ").strip() for l in raw.splitlines() if l.strip()]
    if not statements:
        return 0.0
    ctx = "\n\n".join(contexts)[:2000]
    verdict_raw = _chat(
        _FAITH_JUDGE.format(context=ctx, statements="\n".join(statements)), max_tokens=100
    )
    verdicts = [l.strip().lower() for l in verdict_raw.splitlines() if l.strip()]
    yes_count = sum(1 for v in verdicts if v.startswith("yes"))
    denominator = max(len(verdicts), len(statements))
    return round(yes_count / denominator, 4) if denominator else 0.0

# ---------------------------------------------------------------------------
# METRIC 2 — Answer Relevancy
# Definition (RAGAS paper): ask the LLM to generate N hypothetical questions
# that the answer would answer, then measure average cosine similarity between
# those questions and the original question.  High = answer addresses the query.
# ---------------------------------------------------------------------------
_RELVANCY_PROMPT = """Generate {n} different questions that could be answered by the following text.
Output only the questions, one per line, no numbering or commentary.

Text:
{answer}"""

def answer_relevancy(query: str, answer: str, n: int = 3) -> float:
    raw = _chat(_RELVANCY_PROMPT.format(n=n, answer=answer), max_tokens=200)
    questions = [l.strip() for l in raw.splitlines() if l.strip() and "?" in l][:n]
    if not questions:
        return 0.0
    q_emb = _embed(query).flatten()
    sims  = [float(np.dot(_embed(q).flatten(), q_emb)) for q in questions]
    return round(float(np.mean(sims)), 4)

# ---------------------------------------------------------------------------
# METRIC 3 — Context Precision
# Definition (RAGAS paper): for each retrieved context at rank k, ask the LLM
# whether it is relevant to the question; precision is the mean of a
# rank-weighted score (MAP-style).
# ---------------------------------------------------------------------------
_PRECISION_PROMPT = """Is the following context useful for answering the question?
Reply with exactly one word: "yes" or "no".

Question: {query}

Context:
{context}"""

def context_precision(query: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    hits, total = 0.0, 0.0
    for k, ctx in enumerate(contexts, start=1):
        verdict = _chat(_PRECISION_PROMPT.format(query=query, context=ctx[:600]), max_tokens=5)
        relevant = verdict.lower().startswith("yes")
        if relevant:
            hits += 1
            total += hits / k   # precision@k, only counted when relevant
    denom = sum(1 for _ in contexts)
    return round(total / denom, 4) if denom else 0.0

# ---------------------------------------------------------------------------
# Test questions
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    "What is overfitting and how do you prevent it?",
    "Explain how gradient descent works in neural networks.",
    "What is backpropagation and why is it important?",
    "How do Convolutional Neural Networks detect features in images?",
    "What is the vanishing gradient problem and how do LSTMs solve it?",
    "Explain the difference between bagging and boosting.",
    "What is a support vector machine and how does it find the decision boundary?",
    "How does the attention mechanism in transformers work?",
    "What is batch normalization and why does it help training?",
    "Explain the bias-variance tradeoff in machine learning.",
]

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_evaluation(n_questions: int = len(TEST_QUESTIONS)):
    questions = TEST_QUESTIONS[:n_questions]
    print(f"📋  Evaluating {len(questions)} questions …\n")

    records   = []
    faith_scores, rel_scores, prec_scores = [], [], []

    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q[:70]}")
        t0 = time.time()

        try:
            contexts = retrieve(q)
            answer   = generate(q, contexts)

            f = faithfulness(answer, contexts)
            r = answer_relevancy(q, answer)
            p = context_precision(q, contexts)

            faith_scores.append(f)
            rel_scores.append(r)
            prec_scores.append(p)

            elapsed = round(time.time() - t0, 1)
            print(f"         faithfulness={f:.3f}  answer_relevancy={r:.3f}  context_precision={p:.3f}  ({elapsed}s)")

            records.append({
                "question":          q,
                "answer":            answer,
                "contexts":          contexts,
                "faithfulness":      f,
                "answer_relevancy":  r,
                "context_precision": p,
                "elapsed_s":         elapsed,
            })
        except Exception as e:
            print(f"         ❌  Error: {e}")
            records.append({"question": q, "error": str(e)})

    summary = {
        "timestamp":          datetime.now().isoformat(),
        "model_llm":          LLM_MODEL,
        "model_embed":        EMBED_MODEL,
        "n_questions":        len(questions),
        "faithfulness":       round(float(np.mean(faith_scores)), 4) if faith_scores else None,
        "answer_relevancy":   round(float(np.mean(rel_scores)),   4) if rel_scores   else None,
        "context_precision":  round(float(np.mean(prec_scores)),  4) if prec_scores  else None,
        "per_question":       records,
    }

    # Save JSON
    json_path = BASE_DIR / "eval_results.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n💾  Saved → {json_path}")

    # Save markdown summary
    _write_markdown(summary)

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Faithfulness        {summary['faithfulness']:.3f}" if summary["faithfulness"] is not None else "  Faithfulness        n/a")
    print(f"  Answer Relevancy    {summary['answer_relevancy']:.3f}" if summary["answer_relevancy"] is not None else "  Answer Relevancy    n/a")
    print(f"  Context Precision   {summary['context_precision']:.3f}" if summary["context_precision"] is not None else "  Context Precision   n/a")
    print("=" * 55)
    return summary


def _write_markdown(s: dict):
    fa  = f"{s['faithfulness']:.3f}"       if s["faithfulness"]       is not None else "n/a"
    ar  = f"{s['answer_relevancy']:.3f}"   if s["answer_relevancy"]   is not None else "n/a"
    cp  = f"{s['context_precision']:.3f}"  if s["context_precision"]  is not None else "n/a"

    rows = ""
    for r in s["per_question"]:
        if "error" in r:
            rows += f"| {r['question'][:60]} | ERROR | — | — | — |\n"
        else:
            rows += (
                f"| {r['question'][:60]} "
                f"| {r['faithfulness']:.3f} "
                f"| {r['answer_relevancy']:.3f} "
                f"| {r['context_precision']:.3f} "
                f"| {r['elapsed_s']}s |\n"
            )

    md = f"""# RAG Evaluation Results

> Generated: {s['timestamp']}
> LLM: `{s['model_llm']}` · Embeddings: `{s['model_embed']}`
> Questions evaluated: {s['n_questions']}

## Aggregate Scores

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | **{fa}** | Fraction of answer statements grounded in retrieved context |
| **Answer Relevancy** | **{ar}** | Cosine similarity between question and reverse-generated questions from the answer |
| **Context Precision** | **{cp}** | Fraction of retrieved chunks judged relevant by the LLM |

> Scores range 0–1. Higher is better.
> Metrics follow the [RAGAS paper](https://arxiv.org/abs/2309.15217) definitions, implemented with local Ollama.

## Per-Question Breakdown

| Question | Faithfulness | Answer Relevancy | Context Precision | Time |
|----------|-------------|-----------------|-------------------|------|
{rows}
"""
    md_path = BASE_DIR / "eval_summary.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"📄  Saved → {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline quality")
    parser.add_argument("--questions", type=int, default=len(TEST_QUESTIONS),
                        help=f"Number of test questions (max {len(TEST_QUESTIONS)})")
    args = parser.parse_args()

    if not (1 <= args.questions <= len(TEST_QUESTIONS)):
        sys.exit(f"--questions must be 1–{len(TEST_QUESTIONS)}")

    run_evaluation(args.questions)

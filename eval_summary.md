# RAG Evaluation Results

> Generated: 2026-05-31T19:01:27.504500
> LLM: `llama3.2` · Embeddings: `nomic-embed-text`
> Questions evaluated: 5

## Aggregate Scores

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | **0.773** | Fraction of answer statements grounded in retrieved context |
| **Answer Relevancy** | **0.773** | Cosine similarity between question and reverse-generated questions from the answer |
| **Context Precision** | **0.121** | Fraction of retrieved chunks judged relevant by the LLM |

> Scores range 0–1. Higher is better.
> Metrics follow the [RAGAS paper](https://arxiv.org/abs/2309.15217) definitions, implemented with local Ollama.

## Per-Question Breakdown

| Question | Faithfulness | Answer Relevancy | Context Precision | Time |
|----------|-------------|-----------------|-------------------|------|
| What is overfitting and how do you prevent it? | 0.833 | 0.769 | 0.275 | 84.2s |
| Explain how gradient descent works in neural networks. | 0.714 | 0.752 | 0.151 | 81.9s |
| What is backpropagation and why is it important? | 0.818 | 0.846 | 0.065 | 78.2s |
| How do Convolutional Neural Networks detect features in imag | 0.900 | 0.769 | 0.000 | 83.6s |
| What is the vanishing gradient problem and how do LSTMs solv | 0.600 | 0.730 | 0.112 | 82.7s |


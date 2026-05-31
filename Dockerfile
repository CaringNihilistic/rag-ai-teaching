FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch first — saves ~1.5 GB vs the default CUDA wheel
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# App dependencies (sentence-transformers reuses the torch above)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Source code + static assets only — model files come from S3 at runtime
COPY main.py evaluate.py ./
COPY templates/ templates/
COPY static/    static/

# Render injects $PORT; fall back to 5000 for local docker run
ENV PORT=5000
EXPOSE 5000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-5000}"]

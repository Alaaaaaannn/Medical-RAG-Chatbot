# Medical RAG Chatbot — Hugging Face Spaces (Docker SDK)
FROM python:3.10-slim

# libgomp1 is needed by PyTorch's CPU wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs containers as a non-root user (uid 1000) with a writable home,
# which the model cache needs.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONPATH=/home/user/app \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# Install Python deps first (cached unless requirements.txt changes).
# Drop the editable "-e ." line — at runtime gunicorn adds the app dir to the
# import path, so the local `src` package imports fine without installing it.
COPY --chown=user requirements.txt .
RUN grep -v '^-e' requirements.txt > req.txt \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r req.txt

# Pre-download the embedding + reranker models into the image so the app starts
# quickly and doesn't risk a timeout downloading ~1 GB on first boot.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('BAAI/bge-reranker-base')"

# Copy the app last so code changes don't bust the layers above.
COPY --chown=user . .

EXPOSE 7860

# One worker keeps the models loaded once and the daily budget / session memory
# consistent; a few threads handle concurrent users. Long timeout for slow LLM calls.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "300", \
     "--bind", "0.0.0.0:7860", "app:app"]

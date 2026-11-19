# Agentic RAG: web UI + API in one image.
#   docker build -t agentic-rag .
#   docker run -p 127.0.0.1:8000:8000 -e ANTHROPIC_API_KEY -v rag-data:/data agentic-rag
# Keep the 127.0.0.1: the API has no login, and the Host check doesn't stop other machines (see README).
# Demo mode (no API key, fake model, real search): add -e RAG_FAKE_LLM=1.
# Your own documents: -v /path/to/docs:/app/docs

# ---- frontend
FROM node:22-slim AS frontend
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- app
# Python 3.14: the version requirements.lock was made with.
FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/app \
    FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed \
    RAG_INDEX=/data/index \
    RAG_DATA=/data/app \
    RAG_DOCS=/app/docs \
    RAG_EVALS=/app/evals

RUN useradd --create-home --home-dir /home/app --uid 1000 app \
    && mkdir -p /app /data && chown app:app /app /data
WORKDIR /app

# Exact versions from the lock file (as constraints, so the test tools in it aren't installed).
COPY requirements.txt requirements.lock ./
RUN pip install -r requirements.txt -c requirements.lock

USER app
# Bake in the embedding model and (unless --build-arg BAKE_RERANKER=0) the reranker, so the
# container starts fast and works offline. Without a baked reranker it is downloaded on first
# use, and search works without it if that fails.
ARG BAKE_RERANKER=1
RUN python -c "from chromadb.utils import embedding_functions as e; e.DefaultEmbeddingFunction()(['warm up'])" \
    && if [ "$BAKE_RERANKER" = "1" ]; then python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder as T; T('Xenova/ms-marco-MiniLM-L-6-v2')"; fi

COPY --chown=app:app main.py ./
COPY --chown=app:app rag/ rag/
COPY --chown=app:app server/ server/
COPY --chown=app:app docs/ docs/
COPY --chown=app:app evals/ evals/
COPY --from=frontend --chown=app:app /src/dist frontend/dist

VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Agentic RAG: web UI + API in one image.
#   docker build -t agentic-rag .
#   docker run -p 127.0.0.1:8000:8000 -e ANTHROPIC_API_KEY -v rag-data:/data agentic-rag
# Keep the 127.0.0.1 unless you set RAG_PASSWORD: without it there is no login, and the Host check
# doesn't stop other machines (see README).
# Published for linux/amd64 and linux/arm64 (ghcr.io/<owner>/<repo>:latest).
# Demo mode (no API key, fake model, real search): add -e RAG_FAKE_LLM=1.
# Documents: uploads and Drive syncs are kept in the /data volume (docs/, seeded with the samples on first
# start); or use your own folder with -v /path/to/docs:/app/docs. See docker/entrypoint.sh.

# ---- frontend
# Built on the build machine's own platform: the output is static JS/CSS, the same for every target,
# so a multi-platform build doesn't run npm under emulation.
FROM --platform=$BUILDPLATFORM node:22-slim AS frontend
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
    RAG_EVALS=/app/evals

RUN useradd --create-home --home-dir /home/app --uid 1000 app \
    && mkdir -p /app /data && chown app:app /app /data
WORKDIR /app

# Exact versions from the lock file (as constraints, so the test tools in it aren't installed).
COPY requirements.txt requirements.lock ./
# The chown: emulators can write caches under $HOME while root runs these steps (Docker Desktop's
# Rosetta creates /home/app/.cache/rosetta when building amd64 on Apple Silicon), which would leave
# /home/app/.cache owned by root and break the model download below.
RUN pip install -r requirements.txt -c requirements.lock \
    && chown -R app:app /home/app

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
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/rag-entrypoint

VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/ping', timeout=4)"
ENTRYPOINT ["rag-entrypoint"]
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

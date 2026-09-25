# VKLLM worker container.
# CPU-only (containers can't reach the Mac's MPS GPU). Model weights are
# downloaded from HuggingFace on first startup and cached in the container.

FROM python:3.13-slim

# uv for fast, reproducible installs (matches the dev workflow)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# HF cache lives here; can be mounted as a volume to persist across restarts
ENV HF_HOME=/app/.hf_cache \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# Install dependencies first (layer caching: deps change less often than code)
COPY pyproject.toml uv.lock README.md ./
# Install CPU-only torch explicitly to avoid pulling CUDA wheels (smaller image)
RUN uv pip install --system --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch fastapi "uvicorn[standard]" transformers safetensors

# Now the source
COPY src/ ./src/

EXPOSE 8000

# Liveness check the orchestrator / Podman can read
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Run the worker
CMD ["uvicorn", "vkllm.server:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

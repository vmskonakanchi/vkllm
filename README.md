# VKLLM

A miniature, from-scratch LLM **inference engine** — built to understand how
production serving systems like vLLM actually work, one layer at a time.

Every core piece is hand-written and verified against HuggingFace: the
transformer forward pass, the KV cache, continuous batching, and an HTTP
serving layer. No `model.generate()`, no black boxes — the point is to *own*
the engine end to end.

Runs [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
(a Llama-style model: RoPE, RMSNorm, SwiGLU, grouped-query attention) on CPU or
Apple Silicon (MPS).

## Why

Modern LLM serving is a systems problem: how do you turn a variable amount of
work per request into efficient, batched GPU work while managing the KV cache —
the main memory bottleneck? VKLLM builds the answer incrementally so each
optimization is grounded in the problem it solves.

## Architecture

```
        Client
          │  HTTP
     ┌────────────┐
     │  FastAPI   │   src/vkllm/server.py
     └─────┬──────┘
           │  submit request, await result
     ┌────────────┐
     │ Scheduler  │   src/vkllm/scheduler.py   (continuous batching)
     └─────┬──────┘
    admit → prefill → BATCHED decode → retire
           │
     ┌────────────┐
     │   Model    │   src/vkllm/model.py       (hand-written forward pass)
     └─────┬──────┘
           │
      per-request KV cache
```

## What's implemented

- **Forward pass from scratch** — embedding, RMSNorm, RoPE (Llama half-split
  convention), grouped-query attention (GQA), SwiGLU MLP, residual blocks,
  tied output projection. Verified to match HuggingFace logits to `~1e-4`.
- **KV cache** — prefill/decode split; decode reuses cached keys/values instead
  of recomputing them. Bit-for-bit identical output to the cache-free path.
- **Continuous batching** — a scheduler that admits, batches, and retires
  requests per token step. No head-of-line blocking; short requests finish and
  free their slot while long ones keep running.
- **True tensor-batched decode** — many requests advance in a single batched
  forward pass (left-padded caches + padding mask).
- **Paged KV cache (PagedAttention)** — the cache is split into fixed-size
  blocks from a shared pool, mapped per request by a block table (OS-style
  virtual memory). Removes padding/reallocation waste; freed blocks are reused.
  Verified token-for-token identical to the contiguous path.
- **HTTP server** — FastAPI front door; concurrent requests are batched by a
  background engine loop.
- **Logging** — centralized, level-controlled observability of the request
  lifecycle.

## Roadmap

- [x] Phase 1 — Transformer forward pass (verified vs HuggingFace)
- [x] Phase 2 — Single-request inference
- [x] Phase 3 — KV cache
- [x] Phase 4 — Continuous batching + HTTP server
- [x] Phase 5 — Paged KV cache (PagedAttention)
- [x] Phase 6 — KV-cache-aware scheduling (memory-based admission + recomputation preemption)
- [ ] Phase 7 — Tensor parallelism *(next)*
- [ ] Phase 8 — Distributed inference
- [ ] Phase 9 — Failure handling + observability
- [ ] Phase 10 — GPU-accelerated execution (batched prefill on MPS/CUDA)

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
```

## Run the server

```bash
uv run uvicorn vkllm.server:app --app-dir src
```

Then open the interactive docs at http://127.0.0.1:8000/docs, or:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "The capital of France is", "max_new_tokens": 30}'
```

Set `VKLLM_LOG_LEVEL=DEBUG` to see per-step batch composition.

## Tests

The whole engine is verified against HuggingFace and against the naive
(unoptimized) paths — every optimization is proven to preserve output.

```bash
uv run pytest
```

## Project layout

```
src/vkllm/
  config.py      # model config (loaded from HF config.json)
  model.py       # hand-written forward pass + KV cache + batched/paged decode
  paged_cache.py # block pool + block tables + paged KV storage (PagedAttention)
  scheduler.py   # continuous-batching scheduler + Request state
  server.py      # FastAPI serving layer
  logger.py      # centralized logging
tests/           # pytest suite (component + end-to-end correctness)
```

## Note

This is a learning project prioritizing clarity over raw performance. The goal
is to make every layer of an inference engine legible.

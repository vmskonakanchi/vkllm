"""VKLLM HTTP server -- a production-shaped WORKER node.

Beyond generating text, this exposes what an orchestrator needs to manage a
worker in a cluster:
  GET  /health   liveness  -- am I alive?
  GET  /ready    readiness -- model loaded AND not overloaded -> safe to route to
  GET  /stats    capacity  -- active/waiting/free-blocks/utilization (route by load)
  GET  /metrics  Prometheus-format throughput/latency counters
  POST /generate run inference (batched with other concurrent requests)

Requests submit to a single shared Scheduler running as a background engine loop.

Run:  uv run uvicorn vkllm.server:app --app-dir src
"""

import asyncio
import itertools

import torch
from fastapi import FastAPI, Response
from pydantic import BaseModel

from transformers import AutoTokenizer

from vkllm.logger import get_logger
from vkllm.model import Model
from vkllm.scheduler import Request, Scheduler

MODEL_ID = "HuggingFaceTB/SmolLM-135M"

log = get_logger(__name__)
app = FastAPI(title="VKLLM Worker")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = Model(MODEL_ID)
scheduler = Scheduler(model, max_active=16)

_ids = itertools.count()
_done_events: dict[str, asyncio.Event] = {}
_results: dict[str, Request] = {}

# lifecycle flags for graceful drain
_state = {"ready": False, "draining": False}


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 30


class GenerateResponse(BaseModel):
    text: str
    generated_tokens: int


@app.on_event("startup")
async def start_engine():
    log.info("engine starting (max_active=%d)", scheduler.max_active)
    asyncio.create_task(engine_loop())
    _state["ready"] = True


@app.on_event("shutdown")
async def stop_engine():
    # graceful drain: stop accepting new work, let the loop finish in-flight
    _state["draining"] = True
    log.info("draining: waiting for in-flight requests to finish")
    for _ in range(1000):                      # bounded wait
        if not scheduler.has_work():
            break
        await asyncio.sleep(0.02)
    log.info("drain complete")


async def engine_loop():
    while True:
        if scheduler.has_work():
            scheduler.step()
            while scheduler.finished:
                req = scheduler.finished.pop()
                _results[req.id] = req
                ev = _done_events.get(req.id)
                if ev and not ev.is_set():
                    ev.set()
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(0.005)


# --- worker management endpoints -----------------------------------------

@app.get("/health")
def health():
    """Liveness: is the process up? (K8s livenessProbe)"""
    return {"status": "ok", "model": MODEL_ID}


@app.get("/ready")
def ready():
    """Readiness: loaded, not draining, not overloaded? (K8s readinessProbe)
    An orchestrator should stop routing here if this returns 503."""
    if not _state["ready"] or _state["draining"] or scheduler.is_overloaded():
        reason = ("draining" if _state["draining"]
                  else "overloaded" if scheduler.is_overloaded()
                  else "starting")
        return Response(content=f'{{"ready": false, "reason": "{reason}"}}',
                        media_type="application/json", status_code=503)
    return {"ready": True}


@app.get("/stats")
def stats():
    """Capacity snapshot so an orchestrator can route by load."""
    return scheduler.stats()


@app.get("/metrics")
def metrics():
    """Prometheus-format metrics."""
    return Response(content=scheduler.metrics.prometheus(), media_type="text/plain")


# --- inference ------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
async def generate(req_in: GenerateRequest):
    if _state["draining"]:
        return Response(content='{"error": "server draining"}',
                        media_type="application/json", status_code=503)

    rid = f"req{next(_ids)}"
    ids = torch.tensor(tokenizer(req_in.prompt)["input_ids"])
    request = Request(rid, ids, req_in.max_new_tokens, model)

    event = asyncio.Event()
    _done_events[rid] = event
    scheduler.add_request(request)

    await event.wait()
    _done_events.pop(rid, None)
    result = _results.pop(rid, request)

    if result.failed:
        return Response(content=f'{{"error": "{result.error}"}}',
                        media_type="application/json", status_code=500)

    return GenerateResponse(
        text=tokenizer.decode(result.all_ids.tolist()),
        generated_tokens=len(result.generated),
    )

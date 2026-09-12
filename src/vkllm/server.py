"""VKLLM HTTP server with continuous batching.

Requests arrive concurrently over HTTP, get submitted to a single shared
Scheduler running as a background loop, and are batched together token-by-token.
This is the "HTTP / API" + "Scheduler" boxes wired to the engine.

Run:  uv run uvicorn vkllm.server:app
Then: http://127.0.0.1:8000/docs
"""

import asyncio
import itertools

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from vkllm.logger import get_logger
from vkllm.model import Model
from vkllm.scheduler import Request, Scheduler

MODEL_ID = "HuggingFaceTB/SmolLM-135M"

log = get_logger(__name__)
app = FastAPI(title="VKLLM")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = Model(MODEL_ID)
scheduler = Scheduler(model, max_active=16)

_ids = itertools.count()                 # unique request ids
_done_events: dict[str, asyncio.Event] = {}   # request_id -> "finished" signal


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 30


class GenerateResponse(BaseModel):
    text: str
    generated_tokens: int


@app.on_event("startup")
async def start_engine():
    # Run the scheduler as a background task: tick whenever there's work.
    log.info("engine starting (max_active=%d)", scheduler.max_active)
    asyncio.create_task(engine_loop())


async def engine_loop():
    """The engine heartbeat. Steps the scheduler continuously; when a request
    finishes, signal whoever is awaiting it."""
    while True:
        if scheduler.has_work():
            scheduler.step()
            # signal + drain any requests that just finished
            while scheduler.finished:
                req = scheduler.finished.pop()
                ev = _done_events.get(req.id)
                if ev and not ev.is_set():
                    ev.set()
            await asyncio.sleep(0)       # yield to the event loop each tick
        else:
            await asyncio.sleep(0.005)   # idle: let new requests arrive


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "active": len(scheduler.active)}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req_in: GenerateRequest):
    rid = f"req{next(_ids)}"
    ids = torch.tensor(tokenizer(req_in.prompt)["input_ids"])
    request = Request(rid, ids, req_in.max_new_tokens, model)

    event = asyncio.Event()
    _done_events[rid] = event
    scheduler.add_request(request)

    await event.wait()                   # sleep until the engine finishes this request
    _done_events.pop(rid, None)

    return GenerateResponse(
        text=tokenizer.decode(request.all_ids.tolist()),
        generated_tokens=len(request.generated),
    )

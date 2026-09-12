"""VKLLM scheduler -- the continuous-batching engine ("Scheduler" box).

This file is PLAIN PYTHON systems logic, no tensor math. It tracks per-request
state and orchestrates prefill/decode so many requests share the engine without
head-of-line blocking.

You (the human) fill in the TODOs. The math lives in model.py and stays untouched.
"""

from __future__ import annotations

import torch

from vkllm.logger import get_logger

log = get_logger(__name__)


class Request:
    """State for a single in-flight generation request.

    The scheduler creates one of these per user request and drives it from
    'just arrived' -> prefilled -> decoding -> done.
    """

    def __init__(self, request_id: str, prompt_ids: torch.Tensor, max_new_tokens: int, model):
        self.id = request_id
        self.prompt_ids = prompt_ids          # (prompt_len,) the original prompt tokens
        self.max_new_tokens = max_new_tokens  # how many tokens to generate

        # This request's OWN KV cache (per-request, as in Phase 3).
        self.cache = model.new_cache()

        # Running state:
        self.generated: list[int] = []        # token ids we've generated so far
        self.all_ids = prompt_ids             # prompt + generated, grows each step
        self.prefilled = False                # have we run the prompt through yet?
        self.next_token: torch.Tensor | None = None  # the most recent token to feed next

    # --- small state helpers: YOU implement these (pure logic, no math) ---

    def is_done(self) -> bool:
        # compare len(self.generated) to self.max_new_tokens.
        return len(self.generated) >= self.max_new_tokens

    def position_offset(self) -> int:
        # TODO(you): how many tokens are ALREADY in this request's cache?
        # (Needed for RoPE + masking, same as Phase 3's generate.)
        # Hint: it's the total length so far minus the one token we're about to feed.
        #   during decode, that's len(prompt) + len(generated) - 1
        # For the very first decode step right after prefill, think about what's cached.
        return len(self.prompt_ids) + len(self.generated) - 1

    def record_token(self, token_id: int) -> None:
        self.generated.append(token_id)
        tok = torch.tensor([token_id], device=self.all_ids.device)  # (1,) tensor on right device
        self.all_ids = torch.cat([self.all_ids, tok], dim=0)
        self.next_token = tok



class Scheduler:
    """The continuous-batching engine heartbeat.

    Holds a set of ACTIVE requests. Each 'step' advances every active request
    by ONE token, retires finished ones (freeing their slot), and admits waiting
    ones. No request waits for another to finish -> no head-of-line blocking.

    For this first version we advance requests by looping over them and calling
    the single-sequence forward pass once each. (True tensor-batching -- one
    forward pass over all requests at once -- is the later optimization.)
    """

    def __init__(self, model, max_active: int = 8):
        self.model = model
        self.max_active = max_active          # cap on requests processed concurrently
        self.waiting: list[Request] = []      # arrived but not yet admitted
        self.active: list[Request] = []       # currently being decoded
        self.finished: list[Request] = []     # done, results ready to return

    def add_request(self, req: Request) -> None:
        # A new request arrives -> put it in the waiting queue.
        self.waiting.append(req)
        log.info("request %s arrived (prompt_len=%d, max_new=%d) | waiting=%d",
                 req.id, len(req.prompt_ids), req.max_new_tokens, len(self.waiting))

    def _admit(self) -> None:
        """Move waiting requests into active while there's room."""
        # while there are free slots AND waiting requests, admit the OLDEST
        # (pop(0) = FIFO, fair first-come-first-served).
        while len(self.active) < self.max_active and self.waiting:
            req = self.waiting.pop(0)
            self.active.append(req)
            log.info("request %s admitted | active=%d waiting=%d",
                     req.id, len(self.active), len(self.waiting))

    def _step_request(self, req: Request) -> None:
        """Advance ONE request by ONE token (prefill if needed, else decode)."""
        if not req.prefilled:
            # PREFILL: run the whole prompt through once, fill its cache.
            logits = self.model.forward(req.prompt_ids, cache=req.cache, position_offset=0)
            req.prefilled = True
        else:
            # DECODE: feed only the one new token; cache holds the rest.
            logits = self.model.forward(
                req.next_token, cache=req.cache, position_offset=req.position_offset()
            )
        # greedy pick of the next token from the LAST position's logits
        next_id = int(logits[-1].argmax())
        req.record_token(next_id)

    @torch.no_grad()
    def step(self) -> None:
        """One scheduler tick: admit, advance every active request, retire finished.

        Prefill is done per-request (each prompt is a different length).
        Decode is BATCHED: all already-prefilled requests advance together in
        one model.decode_batch call -> the real throughput win.
        """
        self._admit()                          # 1. pull waiting -> active

        # 2a. PREFILL any request that hasn't been prefilled yet (individually).
        #     Track who we prefill THIS tick so they don't also decode this tick
        #     (that would advance them two tokens in one step).
        just_prefilled = []
        for req in self.active:
            if not req.prefilled:
                logits = self.model.forward(req.prompt_ids, cache=req.cache, position_offset=0)
                req.prefilled = True
                req.record_token(int(logits[-1].argmax()))
                just_prefilled.append(req)
        if just_prefilled:
            log.debug("prefilled %d request(s): %s",
                      len(just_prefilled), [r.id for r in just_prefilled])

        # 2b. BATCHED DECODE for requests prefilled on a PREVIOUS tick, not done.
        decoding = [r for r in self.active
                    if r.prefilled and not r.is_done() and r not in just_prefilled]
        if decoding:
            token_ids = [int(r.next_token) for r in decoding]
            caches = [r.cache for r in decoding]
            offsets = [r.position_offset() for r in decoding]
            next_ids = self.model.decode_batch(token_ids, caches, offsets)   # (B,)
            for r, tid in zip(decoding, next_ids):
                r.record_token(int(tid))
            log.debug("decoded batch of %d | max_offset=%d",
                      len(decoding), max(offsets))

        # 3. retire finished (rebuild list; don't mutate while iterating).
        still_active = []
        for req in self.active:
            if req.is_done():
                self.finished.append(req)
                log.info("request %s finished (%d tokens) | active=%d",
                         req.id, len(req.generated), len(still_active))
            else:
                still_active.append(req)
        self.active = still_active

    def has_work(self) -> bool:
        # anything still to do -> active requests OR waiting requests.
        return len(self.active) > 0 or len(self.waiting) > 0

    def run_until_done(self) -> None:
        """Drive the engine until no work remains (for offline/testing use)."""
        while self.has_work():
            self.step()

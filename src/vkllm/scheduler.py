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
        self.blocks_held = 0                  # KV-cache blocks reserved for this request
        self.preempted = False                # was this request evicted & is resuming?

    def reset_for_resume(self, model) -> None:
        """Preemption: drop the KV cache (free its memory) but KEEP the tokens
        generated so far. On resume the request re-prefills all_ids to rebuild
        the cache -- trading compute (redo prefill) for memory (freed blocks)."""
        self.cache = model.new_cache()        # fresh empty cache; old K/V discarded
        self.prefilled = False                # must re-prefill before decoding
        self.blocks_held = 0                  # blocks were returned to the pool
        self.preempted = True                 # remember so prefill re-runs all_ids

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

    def __init__(self, model, max_active: int = 8,
                 block_size: int = 16, total_blocks: int = 128):
        self.model = model
        self.max_active = max_active          # cap on requests processed concurrently
        # KV-cache MEMORY budget, tracked as a block counter. This models the
        # REAL constraint (memory), separate from the request-count cap.
        self.block_size = block_size
        self.total_blocks = total_blocks
        self.free_blocks = total_blocks       # blocks currently available
        self.waiting: list[Request] = []      # arrived but not yet admitted
        self.active: list[Request] = []       # currently being decoded
        self.finished: list[Request] = []     # done, results ready to return

    def blocks_needed(self, num_tokens: int) -> int:
        """How many fixed-size blocks hold num_tokens (rounded UP)."""
        # ceil(num_tokens / block_size). -(-a // b) is ceiling division, no import.
        return -(-num_tokens // self.block_size)

    def add_request(self, req: Request) -> None:
        # A new request arrives -> put it in the waiting queue.
        self.waiting.append(req)
        log.info("request %s arrived (prompt_len=%d, max_new=%d) | waiting=%d",
                 req.id, len(req.prompt_ids), req.max_new_tokens, len(self.waiting))

    def _admit(self) -> None:
        """Admit waiting requests -- only if there's room in BOTH the active-slot
        cap AND the KV-cache block budget.

        A request needs enough blocks for its prompt to be admitted. If the
        OLDEST waiting request doesn't fit right now, we DEFER it (leave it in
        the queue) rather than crash. This is the core of cache-aware scheduling.
        """
        while self.waiting and len(self.active) < self.max_active:
            req = self.waiting[0]                       # peek at the oldest (FIFO)
            # a resumed (preempted) request re-prefills ALL its tokens so far;
            # a fresh request just needs its prompt.
            need = self.blocks_needed(len(req.all_ids))

            if need > self.free_blocks:
                # Not enough KV memory right now -> defer. Stop here (FIFO: don't
                # skip ahead to a smaller request behind this one).
                log.info("request %s deferred: needs %d blocks, only %d free",
                         req.id, need, self.free_blocks)
                break

            # Fits -> admit and RESERVE its blocks from the budget.
            self.waiting.pop(0)
            self.active.append(req)
            self.free_blocks -= need
            req.blocks_held = need                      # track blocks this req owns
            log.info("request %s admitted%s | reserved %d blocks | free=%d active=%d",
                     req.id, " (resumed)" if req.preempted else "",
                     need, self.free_blocks, len(self.active))

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
                if req.preempted:
                    # RESUME: rebuild the K/V cache for all tokens EXCEPT the
                    # last one, so state matches a request that was never evicted:
                    # cache holds all-but-last, and next_token (the last token)
                    # is decoded normally on the following step. We don't record
                    # a new token here -- we already generated these.
                    self.model.forward(req.all_ids[:-1], cache=req.cache, position_offset=0)
                    req.prefilled = True
                    req.preempted = False
                    req.next_token = req.all_ids[-1:].clone()  # last token, fed next
                else:
                    # FRESH prefill: run the prompt, produce the first token.
                    logits = self.model.forward(req.prompt_ids, cache=req.cache, position_offset=0)
                    req.prefilled = True
                    req.record_token(int(logits[-1].argmax()))
                    just_prefilled.append(req)
                self._reserve_if_grown(req)
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
                self._reserve_if_grown(r)        # each new token may need a block
            log.debug("decoded batch of %d | max_offset=%d",
                      len(decoding), max(offsets))

        # 3. retire finished (rebuild list; don't mutate while iterating).
        #    A finished request RETURNS all its blocks to the budget for reuse.
        still_active = []
        for req in self.active:
            if req.is_done():
                self.free_blocks += req.blocks_held
                req.blocks_held = 0
                self.finished.append(req)
                log.info("request %s finished (%d tokens) | freed blocks | free=%d active=%d",
                         req.id, len(req.generated), self.free_blocks, len(still_active))
            else:
                still_active.append(req)
        self.active = still_active

    def _reserve_if_grown(self, req: Request) -> None:
        """After a request adds a token, reserve one more block if it just
        crossed into a new block. If the budget can't cover it, PREEMPT another
        request to free memory (Layer 2 -- graceful degradation, no crash)."""
        need = self.blocks_needed(len(req.all_ids))
        extra = need - req.blocks_held
        if extra <= 0:
            return                                # still fits in current blocks

        # Free up room by preempting OTHER requests until this token fits.
        while self.free_blocks < extra:
            victim = self._pick_victim(exclude=req)
            if victim is None:
                break                             # nothing left to preempt; proceed anyway
            self._preempt(victim)

        self.free_blocks -= extra
        req.blocks_held = need

    def _pick_victim(self, exclude: Request) -> Request | None:
        """Choose a request to preempt. Policy: the NEWEST active request
        (last admitted) other than `exclude` -- it has waited least and is
        cheapest to redo. Returns None if there's no other request."""
        for r in reversed(self.active):           # newest first
            if r is not exclude and r.blocks_held > 0:
                return r
        return None

    def _preempt(self, victim: Request) -> None:
        """Evict a request: return its blocks to the budget, drop its cache,
        keep its tokens, and requeue it at the FRONT to resume soon."""
        freed = victim.blocks_held
        self.free_blocks += freed
        victim.reset_for_resume(self.model)       # drop K/V, keep tokens
        self.active.remove(victim)
        self.waiting.insert(0, victim)            # front of queue -> resumes soon
        log.info("request %s PREEMPTED | freed %d blocks | free=%d",
                 victim.id, freed, self.free_blocks)

    def has_work(self) -> bool:
        # anything still to do -> active requests OR waiting requests.
        return len(self.active) > 0 or len(self.waiting) > 0

    def run_until_done(self) -> None:
        """Drive the engine until no work remains (for offline/testing use)."""
        while self.has_work():
            self.step()

"""Continuous-batching scheduler: interleaved requests must match single-request output."""

import pytest
import torch

from vkllm.scheduler import Request, Scheduler

SPECS = [
    ("The capital of France is", 30),
    ("Once upon a time", 10),
    ("The meaning of life is", 20),
]


@pytest.fixture(scope="module")
def alone_outputs(model, tokenizer):
    """Ground truth: each request run alone via the verified generate()."""
    out = {}
    for i, (prompt, n) in enumerate(SPECS):
        ids = torch.tensor(tokenizer(prompt)["input_ids"])
        out[i] = model.generate(ids, max_new_tokens=n).tolist()
    return out


def test_scheduler_matches_single_request(model, tokenizer, alone_outputs):
    sched = Scheduler(model, max_active=8)
    reqs = []
    for i, (prompt, n) in enumerate(SPECS):
        ids = torch.tensor(tokenizer(prompt)["input_ids"])
        r = Request(f"req{i}", ids, max_new_tokens=n, model=model)
        reqs.append(r)
        sched.add_request(r)

    sched.run_until_done()

    for i, r in enumerate(reqs):
        assert r.all_ids.tolist() == alone_outputs[i], f"req{i} diverged"


def test_preemption_preserves_output(model, tokenizer, alone_outputs):
    """Under a budget so tight that active requests must be PREEMPTED mid-decode,
    every request must STILL produce identical output (recomputation is lossless)."""
    # block_size 4, only 3 blocks (12 tokens) -> requests WILL overflow while
    # decoding and force preemption/resume cycles.
    sched = Scheduler(model, max_active=8, block_size=4, total_blocks=3)
    reqs = []
    for i, (prompt, n) in enumerate(SPECS):
        ids = torch.tensor(tokenizer(prompt)["input_ids"])
        r = Request(f"req{i}", ids, max_new_tokens=n, model=model)
        reqs.append(r)
        sched.add_request(r)

    sched.run_until_done()   # must survive preemption cycles without corruption

    for i, r in enumerate(reqs):
        assert r.all_ids.tolist() == alone_outputs[i], f"req{i} corrupted by preemption"
    assert sched.free_blocks == sched.total_blocks   # budget balanced after all done


def test_cache_aware_admission_defers_under_pressure(model, tokenizer, alone_outputs):
    """With a TIGHT block budget, requests must be DEFERRED (wait their turn)
    rather than crash -- and every request must still produce correct output."""
    # Tiny budget: block_size 16, only 4 blocks = 64 tokens total capacity.
    # The 3 requests (with generation) can't all fit at once, so some wait.
    sched = Scheduler(model, max_active=8, block_size=16, total_blocks=4)
    reqs = []
    for i, (prompt, n) in enumerate(SPECS):
        ids = torch.tensor(tokenizer(prompt)["input_ids"])
        r = Request(f"req{i}", ids, max_new_tokens=n, model=model)
        reqs.append(r)
        sched.add_request(r)

    sched.run_until_done()   # must complete, not deadlock/crash

    # all requests still correct despite being scheduled under memory pressure
    for i, r in enumerate(reqs):
        assert r.all_ids.tolist() == alone_outputs[i], f"req{i} diverged"
    # budget fully returned after everything finishes
    assert sched.free_blocks == sched.total_blocks

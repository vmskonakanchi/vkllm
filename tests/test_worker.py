"""Worker-node features: metrics, capacity stats, failure isolation."""

import torch

from vkllm.metrics import Metrics
from vkllm.scheduler import Request, Scheduler


def test_metrics_percentiles_and_throughput():
    m = Metrics()
    m.record_arrival()
    m.record_completion(latency_s=0.1, tokens=10)
    m.record_completion(latency_s=0.3, tokens=20)
    snap = m.snapshot()
    assert snap["requests_completed"] == 2
    assert snap["tokens_generated"] == 30
    assert snap["latency_p50_ms"] > 0
    assert "vkllm_tokens_generated 30" in m.prometheus()


def test_metrics_recorded_through_scheduler(model, tokenizer):
    sched = Scheduler(model, max_active=8)
    ids = torch.tensor(tokenizer("The capital of France is")["input_ids"])
    sched.add_request(Request("r0", ids, max_new_tokens=8, model=model))
    sched.run_until_done()
    snap = sched.metrics.snapshot()
    assert snap["requests_total"] == 1
    assert snap["requests_completed"] == 1
    assert snap["tokens_generated"] == 8
    assert snap["requests_failed"] == 0


def test_stats_reports_capacity(model, tokenizer):
    sched = Scheduler(model, max_active=8, block_size=16, total_blocks=64)
    s0 = sched.stats()
    assert s0["blocks_free"] == 64 and s0["active"] == 0
    assert s0["kv_utilization"] == 0.0

    ids = torch.tensor(tokenizer("Hello world")["input_ids"])
    sched.add_request(Request("r0", ids, max_new_tokens=5, model=model))
    sched.step()                       # admit + prefill -> consumes blocks
    s1 = sched.stats()
    assert s1["active"] == 1
    assert s1["blocks_free"] < 64      # some reserved
    assert 0.0 < s1["kv_utilization"] <= 1.0


def test_failure_isolation_does_not_crash_engine(model, tokenizer):
    """A request that errors is retired as FAILED; other requests still complete."""
    sched = Scheduler(model, max_active=8)

    good = Request("good", torch.tensor(tokenizer("The capital of France is")["input_ids"]),
                   max_new_tokens=8, model=model)
    # a poisoned request: negative token id will blow up the embedding lookup
    bad = Request("bad", torch.tensor([-999999]), max_new_tokens=8, model=model)

    sched.add_request(good)
    sched.add_request(bad)
    sched.run_until_done()             # must NOT raise

    assert good.failed is False
    assert len(good.generated) == 8    # good request completed normally
    assert bad.failed is True          # bad request isolated as failed
    assert sched.metrics.snapshot()["requests_failed"] == 1
    assert sched.free_blocks == sched.total_blocks   # blocks reclaimed even on failure

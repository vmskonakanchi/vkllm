"""Metrics collection for the VKLLM worker.

Tracks the numbers an orchestrator (and you) need to see: throughput, latency
percentiles, request counts, and errors. This is the observability layer that
turns a black-box engine into a monitorable worker.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class Metrics:
    """Thread-safe metrics collector.

    Latencies are kept in a bounded ring buffer so percentiles reflect RECENT
    behavior and memory stays constant (you don't keep every latency forever).
    """

    def __init__(self, window: int = 1000):
        self._lock = threading.Lock()
        self.started_at = time.time()

        # counters (monotonic)
        self.requests_total = 0
        self.requests_failed = 0
        self.requests_completed = 0
        self.tokens_generated = 0

        # recent latencies (seconds), bounded so percentiles are O(window)
        self._latencies: deque[float] = deque(maxlen=window)

    # --- recording (called by the engine) ---------------------------------

    def record_arrival(self) -> None:
        with self._lock:
            self.requests_total += 1

    def record_completion(self, latency_s: float, tokens: int) -> None:
        with self._lock:
            self.requests_completed += 1
            self.tokens_generated += tokens
            self._latencies.append(latency_s)

    def record_failure(self) -> None:
        with self._lock:
            self.requests_failed += 1

    # --- reporting (called by /metrics, /stats) ---------------------------

    @staticmethod
    def _percentile(sorted_vals: list[float], pct: float) -> float:
        if not sorted_vals:
            return 0.0
        # nearest-rank percentile
        k = max(0, min(len(sorted_vals) - 1, int(round(pct / 100 * (len(sorted_vals) - 1)))))
        return sorted_vals[k]

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self._latencies)
            uptime = max(1e-9, time.time() - self.started_at)
            return {
                "uptime_s": round(uptime, 1),
                "requests_total": self.requests_total,
                "requests_completed": self.requests_completed,
                "requests_failed": self.requests_failed,
                "tokens_generated": self.tokens_generated,
                "throughput_tokens_per_s": round(self.tokens_generated / uptime, 2),
                "latency_p50_ms": round(self._percentile(lat, 50) * 1000, 1),
                "latency_p99_ms": round(self._percentile(lat, 99) * 1000, 1),
                "latency_samples": len(lat),
            }

    def prometheus(self) -> str:
        """Render the snapshot in Prometheus text exposition format."""
        s = self.snapshot()
        lines = [
            "# HELP vkllm_requests_total Total requests received",
            "# TYPE vkllm_requests_total counter",
            f"vkllm_requests_total {s['requests_total']}",
            "# TYPE vkllm_requests_completed counter",
            f"vkllm_requests_completed {s['requests_completed']}",
            "# TYPE vkllm_requests_failed counter",
            f"vkllm_requests_failed {s['requests_failed']}",
            "# TYPE vkllm_tokens_generated counter",
            f"vkllm_tokens_generated {s['tokens_generated']}",
            "# TYPE vkllm_throughput_tokens_per_second gauge",
            f"vkllm_throughput_tokens_per_second {s['throughput_tokens_per_s']}",
            "# TYPE vkllm_latency_p50_ms gauge",
            f"vkllm_latency_p50_ms {s['latency_p50_ms']}",
            "# TYPE vkllm_latency_p99_ms gauge",
            f"vkllm_latency_p99_ms {s['latency_p99_ms']}",
        ]
        return "\n".join(lines) + "\n"

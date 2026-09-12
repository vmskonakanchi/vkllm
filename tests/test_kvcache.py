"""KV cache correctness: cached generation must match cache-free generation."""

import torch


def test_cached_matches_cache_free(model, tokenizer):
    ids = torch.tensor(tokenizer("The capital of France is")["input_ids"])
    slow = model.generate_slow(ids, max_new_tokens=40)
    fast = model.generate(ids, max_new_tokens=40)
    assert bool((slow == fast).all())

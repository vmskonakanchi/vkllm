"""True tensor-batched decode must match single-request decode."""

import torch

SPECS = [
    ("The capital of France is", 30),
    ("Once upon a time", 30),
    ("The meaning of life is", 30),
]


def _batched_generate(model, prompts, n_tokens):
    """Prefill each prompt individually, then batched-decode all together."""
    caches, offsets, next_toks, results = [], [], [], []
    for ids in prompts:
        cache = model.new_cache()
        logits = model.forward(ids, cache=cache, position_offset=0)
        first = int(logits[-1].argmax())
        caches.append(cache)
        offsets.append(ids.shape[0])
        next_toks.append(first)
        results.append(ids.tolist() + [first])

    for _ in range(n_tokens - 1):
        nxt = model.decode_batch(next_toks, caches, offsets)
        for b in range(len(prompts)):
            tid = int(nxt[b])
            results[b].append(tid)
            next_toks[b] = tid
            offsets[b] += 1
    return results


def test_batched_decode_matches_single(model, tokenizer):
    prompts = [torch.tensor(tokenizer(p)["input_ids"]) for p, _ in SPECS]
    n = SPECS[0][1]

    alone = [model.generate(p, max_new_tokens=n).tolist() for p in prompts]
    batched = _batched_generate(model, prompts, n)

    for i in range(len(prompts)):
        assert batched[i] == alone[i], f"req{i} diverged in batched decode"

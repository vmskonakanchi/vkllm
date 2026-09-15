"""Paged KV cache: allocator/block-table logic + paged generation == contiguous."""

import torch

from vkllm.paged_cache import BlockPool, BlockTable, PagedKVCache


def test_block_pool_allocate_free_reuse():
    pool = BlockPool(num_blocks=4, block_size=16)
    assert pool.num_free() == 4
    a, b, c, d = pool.allocate(), pool.allocate(), pool.allocate(), pool.allocate()
    assert pool.num_free() == 0
    try:
        pool.allocate()
        assert False, "should have raised when exhausted"
    except RuntimeError:
        pass
    pool.free_many([a, b])
    assert pool.num_free() == 2
    assert pool.allocate() in (a, b)   # reuses a freed block


def test_block_table_crosses_boundaries_and_frees():
    pool = BlockPool(num_blocks=10, block_size=4)
    bt = BlockTable(pool)
    placements = [bt.append_token() for _ in range(9)]  # 9 tokens, block_size 4
    assert len(bt.block_ids) == 3                       # 4 + 4 + 1
    assert placements[3][1] == 3                        # last slot of first block
    assert placements[4][1] == 0                        # first slot of second block
    assert placements[4][0] != placements[3][0]         # a new block
    bt.free()
    assert pool.num_free() == 10                        # all returned


def test_paged_gather_equals_contiguous():
    torch.manual_seed(0)
    kv_heads, head_dim, bs = 3, 64, 4
    cache = PagedKVCache(1, num_blocks=10, block_size=bs, kv_heads=kv_heads, head_dim=head_dim)
    bt = cache.new_block_table()
    ref_k, ref_v = [], []
    for _ in range(9):
        k = torch.randn(kv_heads, 1, head_dim)
        v = torch.randn(kv_heads, 1, head_dim)
        ref_k.append(k); ref_v.append(v)
        cache.append(0, bt, [bt.append_token()], k, v)
    gk, gv = cache.gather(0, bt)
    assert torch.allclose(gk, torch.cat(ref_k, dim=1))
    assert torch.allclose(gv, torch.cat(ref_v, dim=1))


def test_paged_generation_matches_contiguous(model, tokenizer):
    """The end-to-end proof: paged generation == verified generate(), token-for-token."""
    ids = torch.tensor(tokenizer("The capital of France is")["input_ids"])
    n = 30

    reference = model.generate(ids, max_new_tokens=n).tolist()

    paged = PagedKVCache(
        num_layers=model.config.num_layers,
        num_blocks=256,
        block_size=16,
        kv_heads=model.config.num_kv_heads,
        head_dim=model.config.head_dim,
        device=model.device,
    )
    block_table = paged.new_block_table()
    out = model.generate_paged(ids, max_new_tokens=n, paged=paged, block_table=block_table).tolist()

    assert out == reference

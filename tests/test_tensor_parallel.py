"""Tensor parallelism: sharded MLP/attention must equal the single-device result."""

import torch

from vkllm.tensor_parallel import TensorParallel, all_reduce

IDS = torch.tensor([1, 42, 100, 7, 9])


def test_all_reduce_sums_partials():
    a = torch.ones(3)
    b = torch.ones(3) * 2
    c = torch.ones(3) * 4
    assert torch.equal(all_reduce([a, b, c]), torch.tensor([7.0, 7.0, 7.0]))


def test_sharded_mlp_equals_single_device(model):
    """gate/up column-split + down row-split + all-reduce == unsharded mlp()."""
    h = model.embed(IDS)
    normed = model.rmsnorm(h, model.w("model.layers.0.post_attention_layernorm.weight"))

    single = model.mlp(normed, layer=0)

    for n in (2, 3):   # 1536 divides by both
        tp = TensorParallel(model, n_shards=n)
        sharded = tp.mlp(normed, layer=0)
        assert torch.allclose(single, sharded, atol=1e-5), f"MLP mismatch at n_shards={n}"


def test_sharded_attention_equals_single_device(model):
    """heads split across shards + o_proj row-split + all-reduce == unsharded attention()."""
    h = model.embed(IDS)
    normed = model.rmsnorm(h, model.w("model.layers.0.input_layernorm.weight"))

    single = model.attention(normed, layer=0)

    # n_shards must divide BOTH num_q_heads (9) and num_kv_heads (3) -> n=3.
    tp = TensorParallel(model, n_shards=3)
    sharded = tp.attention(normed, layer=0)
    assert torch.allclose(single, sharded, atol=1e-5)


def test_tp_full_block_equals_single_device(model):
    """A full block built from sharded MLP + attention == the single-device block."""
    h = model.embed(IDS)
    tp = TensorParallel(model, n_shards=3)

    # single-device block (same structure as Model.block)
    w_in = model.w("model.layers.0.input_layernorm.weight")
    w_post = model.w("model.layers.0.post_attention_layernorm.weight")
    single = h + model.attention(model.rmsnorm(h, w_in), 0)
    single = single + model.mlp(model.rmsnorm(single, w_post), 0)

    # tensor-parallel block
    tp_out = h + tp.attention(model.rmsnorm(h, w_in), 0)
    tp_out = tp_out + tp.mlp(model.rmsnorm(tp_out, w_post), 0)

    assert torch.allclose(single, tp_out, atol=1e-5)

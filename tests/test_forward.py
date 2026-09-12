"""Forward-pass correctness: each component + the full pass vs HuggingFace."""

import torch

IDS = torch.tensor([1, 42, 100, 7, 9])   # 5 arbitrary token ids


def test_embed(model):
    h = model.embed(IDS)
    assert tuple(h.shape) == (5, model.config.hidden_size)
    assert h.dtype == torch.float32


def test_rmsnorm_unit_rms(model):
    h = model.embed(IDS)
    w_norm = model.w("model.layers.0.input_layernorm.weight")
    normed = model.rmsnorm(h, w_norm)
    assert tuple(normed.shape) == (5, model.config.hidden_size)

    # dividing out the learned weight should leave unit RMS per token
    rms = (normed / w_norm).pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_matches_hf(model):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    cfg = model.config
    q = torch.randn(cfg.num_q_heads, 5, cfg.head_dim)

    cos, sin = model.rope_tables(5)
    q_ours = model.apply_rope(q, cos, sin)

    q_hf, _ = apply_rotary_pos_emb(
        q.unsqueeze(0), q.unsqueeze(0), cos.unsqueeze(0), sin.unsqueeze(0)
    )
    assert torch.allclose(q_ours, q_hf.squeeze(0), atol=1e-5)


def test_attention_shape_and_effect(model):
    h = model.embed(IDS)
    normed = model.rmsnorm(h, model.w("model.layers.0.input_layernorm.weight"))
    out = model.attention(normed, layer=0)
    assert tuple(out.shape) == (5, model.config.hidden_size)
    assert not torch.allclose(out, normed)


def test_mlp_shape_and_effect(model):
    h = model.embed(IDS)
    normed = model.rmsnorm(h, model.w("model.layers.0.post_attention_layernorm.weight"))
    out = model.mlp(normed, layer=0)
    assert tuple(out.shape) == (5, model.config.hidden_size)
    assert not torch.allclose(out, normed)


def test_block_shape_and_effect(model):
    h = model.embed(IDS)
    out = model.block(h, layer=0)
    assert tuple(out.shape) == (5, model.config.hidden_size)
    assert not torch.allclose(out, h)


def test_full_forward_matches_hf(model):
    """The proof: full forward pass must match HF's logits (both float32)."""
    from transformers import AutoModelForCausalLM

    ours = model.forward(IDS)

    hf = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", torch_dtype=torch.float32)
    hf.eval()
    with torch.no_grad():
        hf_logits = hf(IDS.unsqueeze(0)).logits[0]

    assert ours.shape == hf_logits.shape
    assert torch.allclose(ours, hf_logits, atol=1e-3)
    assert bool((ours.argmax(-1) == hf_logits.argmax(-1)).all())

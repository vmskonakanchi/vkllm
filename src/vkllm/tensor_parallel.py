"""Tensor parallelism (TP) -- split each layer's weight matrices across N shards.

A model too big for one device is split so each "device" (here: simulated shard,
all on one machine) holds a SLICE of every weight matrix and computes its slice
in parallel. Results are combined with an all-reduce (sum).

Two split styles (weights are stored (out, in); a bias-free Linear is x @ W.T):
  * COLUMN split  = split the OUTPUT dim -> split W along dim 0 (rows of the
    stored matrix). Each shard produces PART of the output. Combine = concat.
  * ROW split     = split the INPUT dim  -> split W along dim 1 (cols of the
    stored matrix) AND split the input. Each shard produces a PARTIAL SUM.
    Combine = all-reduce (add).

The TP trick: pair COLUMN-split then ROW-split so only ONE all-reduce is needed
per block.
  - MLP:       gate/up = column-split, down = row-split -> 1 all-reduce.
  - Attention: heads   = column-split (heads are independent!), o_proj =
               row-split -> 1 all-reduce.

Concepts are real; on one machine we run the shards sequentially, so there's no
actual speedup -- but the sharding + all-reduce logic is exactly the real thing.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from vkllm.logger import get_logger

log = get_logger(__name__)


def all_reduce(partials: list[torch.Tensor]) -> torch.Tensor:
    """Combine partial results from all shards by SUMMING them.

    On real hardware this is a network collective where every device ends up
    with the total. On one machine, summing the tensors IS the operation.
    """
    out = partials[0].clone()
    for p in partials[1:]:
        out = out + p
    return out


class TensorParallel:
    """Runs a layer's MLP and attention split across `n_shards` shards,
    validating that the sharded result equals the single-device result.
    """

    def __init__(self, model, n_shards: int = 2):
        self.model = model
        self.cfg = model.config
        self.n = n_shards

    # ------------------------------------------------------------------
    # Sharded MLP:  down( silu(gate(x)) * up(x) )
    #   gate/up : COLUMN-split (each shard computes its slice of the 1536 hidden)
    #   down    : ROW-split    (each shard produces a partial 576 output)
    #   combine : one all-reduce (sum)
    # ------------------------------------------------------------------
    def mlp(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        p = f"model.layers.{layer}.mlp"
        Wg = self.model.w(f"{p}.gate_proj.weight")   # (inter, hidden)
        Wu = self.model.w(f"{p}.up_proj.weight")     # (inter, hidden)
        Wd = self.model.w(f"{p}.down_proj.weight")   # (hidden, inter)

        inter = Wg.shape[0]                           # 1536
        assert inter % self.n == 0, "intermediate size must divide by n_shards"
        shard = inter // self.n

        partials = []
        for s in range(self.n):
            lo, hi = s * shard, (s + 1) * shard
            # COLUMN-split gate/up: this shard owns hidden units [lo:hi].
            Wg_s = Wg[lo:hi]                          # (shard, hidden)
            Wu_s = Wu[lo:hi]
            gate_s = x @ Wg_s.T                       # (seq, shard)
            up_s = x @ Wu_s.T                         # (seq, shard)
            gated_s = F.silu(gate_s) * up_s           # (seq, shard)
            # ROW-split down: this shard owns input cols [lo:hi] of down_proj.
            Wd_s = Wd[:, lo:hi]                       # (hidden, shard)
            partial = gated_s @ Wd_s.T                # (seq, hidden) PARTIAL
            partials.append(partial)

        # combine the partial sums -> full output
        return all_reduce(partials)                   # (seq, hidden)

    # ------------------------------------------------------------------
    # Sharded attention:
    #   q/k/v : split the HEADS across shards (heads are independent)
    #   o_proj: ROW-split (each shard produces a partial output)
    #   combine: one all-reduce (sum)
    #
    # Assumes num_q_heads and num_kv_heads both divide by n_shards.
    # ------------------------------------------------------------------
    def attention(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        cfg = self.cfg
        seq_len = x.shape[0]
        p = f"model.layers.{layer}.self_attn"
        Wq = self.model.w(f"{p}.q_proj.weight")
        Wk = self.model.w(f"{p}.k_proj.weight")
        Wv = self.model.w(f"{p}.v_proj.weight")
        Wo = self.model.w(f"{p}.o_proj.weight")       # (hidden, num_q_heads*head_dim)

        hd = cfg.head_dim
        q_per = cfg.num_q_heads // self.n             # query heads per shard
        kv_per = cfg.num_kv_heads // self.n           # kv heads per shard
        rep = cfg.num_q_heads // cfg.num_kv_heads     # GQA repeat factor
        cos, sin = self.model.rope_tables(seq_len)

        partials = []
        for s in range(self.n):
            # This shard owns query heads [q_lo:q_hi] and kv heads [kv_lo:kv_hi].
            q_lo, q_hi = s * q_per, (s + 1) * q_per
            kv_lo, kv_hi = s * kv_per, (s + 1) * kv_per

            # COLUMN-split q/k/v: slice the projection rows for THIS shard's heads.
            Wq_s = Wq[q_lo * hd:q_hi * hd]            # (q_per*hd, hidden)
            Wk_s = Wk[kv_lo * hd:kv_hi * hd]
            Wv_s = Wv[kv_lo * hd:kv_hi * hd]

            q = (x @ Wq_s.T).view(seq_len, q_per, hd).transpose(0, 1)    # (q_per, seq, hd)
            k = (x @ Wk_s.T).view(seq_len, kv_per, hd).transpose(0, 1)   # (kv_per, seq, hd)
            v = (x @ Wv_s.T).view(seq_len, kv_per, hd).transpose(0, 1)

            q = self.model.apply_rope(q, cos, sin)
            k = self.model.apply_rope(k, cos, sin)

            k = k.repeat_interleave(rep, dim=0)       # GQA expand within the shard
            v = v.repeat_interleave(rep, dim=0)

            scores = q @ k.transpose(-2, -1) / math.sqrt(hd)            # (q_per, seq, seq)
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool,
                                         device=self.model.device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            w = torch.softmax(scores, dim=-1)
            out = w @ v                                                 # (q_per, seq, hd)
            out = out.transpose(0, 1).reshape(seq_len, q_per * hd)      # (seq, q_per*hd)

            # ROW-split o_proj: this shard owns o_proj columns for its heads.
            Wo_s = Wo[:, q_lo * hd:q_hi * hd]         # (hidden, q_per*hd)
            partial = out @ Wo_s.T                     # (seq, hidden) PARTIAL
            partials.append(partial)

        return all_reduce(partials)                    # (seq, hidden)

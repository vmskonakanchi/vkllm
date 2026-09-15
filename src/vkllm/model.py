"""VKLLM hand-written forward pass for SmolLM (Llama-style) models.

We own this forward pass so later phases (KV cache, batching, paging) can
reach inside attention and control exactly how it runs.
"""

from torch._tensor import Tensor
import torch
import math
from vkllm.config import ModelConfig
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from typing import Any

from vkllm.logger import get_logger

_log = get_logger(__name__)


class Model:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.device = device
        self.config = ModelConfig.from_pretrained(model_id)
        _log.info("loading weights for %s on device=%s", model_id, device)
        self.weights = self.load_weights(model_id)
        _log.info("model ready: %d layers, hidden=%d, %d weight tensors",
                  self.config.num_layers, self.config.hidden_size, len(self.weights))

    
    def load_weights(self,model_id:str) -> dict:
        path = hf_hub_download(model_id, "model.safetensors")

        weights: dict[Any, Any] = {}
        with safe_open(path, framework="pt", device=self.device) as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        return weights


    def w(self, name: str) -> torch.Tensor:
        """Convenience: fetch a weight tensor by name."""
        return self.weights[name]

    # ------------------------------------------------------------------
    # Step 3: Embedding
    # token ids (seq_len,) -> hidden vectors (seq_len, hidden_size)
    # ------------------------------------------------------------------
    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Look up rows of model.embed_tokens.weight for each id.
        return self.w("model.embed_tokens.weight")[token_ids]

    # ------------------------------------------------------------------
    # Step 4: RMSNorm
    # Rescales each token's vector to a stable size, then applies a learned
    # per-feature scale. No mean-subtraction, no bias (that's what makes it
    # RMSNorm rather than LayerNorm).
    #   rms   = sqrt( mean(x^2 over hidden dim) + eps )
    #   out   = (x / rms) * weight
    # ------------------------------------------------------------------
    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, hidden_size)   weight: (hidden_size,)

        # 1. square every element, then average across the hidden dim (last axis).
        #    keepdim=True -> shape (seq_len, 1) so it broadcasts back over x.
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)          # (seq_len, 1)

        # 2. root-mean-square, with eps inside the sqrt for numerical safety.
        rms = torch.sqrt(mean_sq + self.config.rms_norm_eps)   # (seq_len, 1)

        # 3. normalize each vector to unit RMS, then apply the learned scale.
        return (x / rms) * weight                              # (seq_len, hidden_size)

    # ------------------------------------------------------------------
    # Step 5: RoPE (Rotary Position Embedding)
    #
    # Injects position information by ROTATING each query/key vector by an
    # angle that depends on its position. No learned weights -- pure math.
    #
    # Two functions:
    #   rope_tables(seq_len) -> (cos, sin) angle tables, precomputed once.
    #   apply_rope(x, cos, sin) -> rotates x (a Q or K tensor).
    #
    # We use the LLAMA / HF "half-split" convention (pair dim i with dim
    # i + head_dim/2), NOT the interleaved convention. This is the detail
    # that makes hand-written RoPE match HF.
    # ------------------------------------------------------------------
    def rope_tables(self, seq_len: int, position_offset: int = 0):
        head_dim = self.config.head_dim
        theta = self.config.rope_theta

        # 1. Frequency for each of the head_dim/2 pairs.
        #    inv_freq[i] = 1 / theta^(2i / head_dim),  i = 0, 1, ... head_dim/2 - 1
        #    Low i -> high frequency (fast rotation, fine/local position).
        #    High i -> low frequency (slow rotation, coarse/long-range position).
        i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=self.device)  # (head_dim/2,)
        inv_freq = 1.0 / (theta ** (i / head_dim))                # (head_dim/2,)

        # 2. Angle for every (position, pair) = position * inv_freq.
        #    position_offset shifts the positions so a decode token cached at
        #    index N is rotated by angle N (its true position), not 0.
        positions = torch.arange(
            position_offset, position_offset + seq_len,
            dtype=torch.float32, device=self.device,
        )                                                          # (seq_len,)
        angles = torch.outer(positions, inv_freq)                 # (seq_len, head_dim/2)

        # 3. HF half-split: the full head_dim table is the half-table
        #    DUPLICATED (concatenated with itself), so dim i and dim i+half
        #    share the same angle. This pairs first-half with second-half.
        angles = torch.cat([angles, angles], dim=-1)              # (seq_len, head_dim)

        return torch.cos(angles), torch.sin(angles)               # each (seq_len, head_dim)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        # Splits x in half along the last dim and rotates:
        #   [a, b] -> [-b, a]
        # This is the half-split partner of the 2D rotation (x0' = x0*cos - x1*sin).
        half = x.shape[-1] // 2
        first, second = x[..., :half], x[..., half:]
        return torch.cat([-second, first], dim=-1)

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (num_heads, seq_len, head_dim)   cos/sin: (seq_len, head_dim)
        # The 2D rotation, expressed for the whole vector at once:
        #   out = x * cos + rotate_half(x) * sin
        # cos/sin broadcast across the head dimension.
        return x * cos + self.rotate_half(x) * sin

    # ------------------------------------------------------------------
    # Step 6: Attention (one layer) with GQA
    #
    # x is already RMSNorm'd by the caller.  x: (seq_len, hidden_size)
    # Returns: (seq_len, hidden_size)
    #
    # layer is the layer index (0..29) so we fetch the right weights.
    # ------------------------------------------------------------------
    def attention(self, x: torch.Tensor, layer: int,
                  cache: dict | None = None, position_offset: int = 0) -> torch.Tensor:
        # x: (seq_len, hidden_size) -- the NEW tokens only (whole prompt in
        # prefill, or a single token in decode).
        # cache: this layer's {"k": ..., "v": ...} of PAST tokens, or None.
        # position_offset: how many tokens are already cached (for RoPE).
        cfg = self.config
        seq_len = x.shape[0]
        p = f"model.layers.{layer}.self_attn"

        Wq = self.w(f"{p}.q_proj.weight")   # (576, 576)
        Wk = self.w(f"{p}.k_proj.weight")   # (192, 576)
        Wv = self.w(f"{p}.v_proj.weight")   # (192, 576)
        Wo = self.w(f"{p}.o_proj.weight")   # (576, 576)

        # --- 1. project to q, k, v (for the NEW tokens) ------------------
        q = x @ Wq.T
        k = x @ Wk.T
        v = x @ Wv.T

        # --- 2. split into heads -----------------------------------------
        q = q.view(seq_len, cfg.num_q_heads, cfg.head_dim).transpose(0, 1)    # (9, seq, 64)
        k = k.view(seq_len, cfg.num_kv_heads, cfg.head_dim).transpose(0, 1)   # (3, seq, 64)
        v = v.view(seq_len, cfg.num_kv_heads, cfg.head_dim).transpose(0, 1)   # (3, seq, 64)

        # --- 3. apply RoPE to q and k (NOT v), at the TRUE positions ------
        cos, sin = self.rope_tables(seq_len, position_offset)
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)

        # --- 3b. KV CACHE: append new k/v, then read the FULL history ----
        # We cache the 3 kv-heads (NOT the expanded 9) -- that's the memory win.
        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=1)   # (3, past+seq, 64)
                v = torch.cat([cache["v"], v], dim=1)
            cache["k"] = k                              # store the grown k/v
            cache["v"] = v
        # after this, k/v span ALL tokens so far; q spans only the new tokens.
        total_len = k.shape[1]                          # past + new

        # --- 4. GQA: expand kv heads 3 -> 9 (transient, not cached) ------
        rep = cfg.num_q_heads // cfg.num_kv_heads
        k = k.repeat_interleave(rep, dim=0)   # (9, total_len, 64)
        v = v.repeat_interleave(rep, dim=0)   # (9, total_len, 64)

        # --- 5. scaled dot-product attention -----------------------------
        # scores: (9, seq, total_len) -- each NEW query row vs ALL keys.
        scores = q @ k.transpose(-2, -1) / math.sqrt(cfg.head_dim)   # (9, seq, total_len)

        # Causal mask: only needed when processing >1 new token (prefill).
        # A single decode token legitimately attends to the whole cache, so
        # no mask. For prefill, new query i (true pos offset+i) may only see
        # keys up to that position.
        if seq_len > 1:
            q_pos = torch.arange(seq_len, device=self.device).unsqueeze(1) + position_offset   # (seq, 1)
            k_pos = torch.arange(total_len, device=self.device).unsqueeze(0)                   # (1, total_len)
            mask = k_pos > q_pos                                          # (seq, total_len)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        out = weights @ v                                            # (9, seq, 64)

        # --- 6. merge heads + output projection --------------------------
        out = out.transpose(0, 1).reshape(seq_len, cfg.hidden_size)   # (seq, 576)
        out = out @ Wo.T                                              # (seq, 576)
        return out

    # ------------------------------------------------------------------
    # Step 7: SwiGLU MLP (one layer)
    # Each token processes itself (no cross-token mixing here).
    #   mlp(x) = down( silu(gate(x)) * up(x) )
    # ------------------------------------------------------------------
    def mlp(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        # x: (seq_len, hidden_size), already RMSNorm'd by the caller
        p = f"model.layers.{layer}.mlp"
        Wg = self.w(f"{p}.gate_proj.weight")   # (1536, 576)
        Wu = self.w(f"{p}.up_proj.weight")     # (1536, 576)
        Wd = self.w(f"{p}.down_proj.weight")   # (576, 1536)

        gate = x @ Wg.T                        # (seq, 1536)  widen
        up = x @ Wu.T                          # (seq, 1536)  widen
        gated = F.silu(gate) * up              # (seq, 1536)  gate the signal
        return gated @ Wd.T                    # (seq, 576)   squeeze back

    # ------------------------------------------------------------------
    # Step 8: One transformer block (pre-norm + residuals)
    #   x = x + attention(rmsnorm(x, input_layernorm))
    #   x = x + mlp(rmsnorm(x, post_attention_layernorm))
    # The `x +` are the RESIDUAL connections: each sublayer learns a change
    # to ADD, and the raw x flows straight through (highway across 30 layers).
    # ------------------------------------------------------------------
    def block(self, x: torch.Tensor, layer: int,
              cache: dict | None = None, position_offset: int = 0) -> torch.Tensor:
        # x: (seq_len, hidden_size)
        w_in = self.w(f"model.layers.{layer}.input_layernorm.weight")
        w_post = self.w(f"model.layers.{layer}.post_attention_layernorm.weight")

        # residual 1: normalize -> attention (cache-aware) -> add back to x
        x = x + self.attention(self.rmsnorm(x, w_in), layer, cache, position_offset)
        # residual 2: normalize -> mlp -> add back to x
        x = x + self.mlp(self.rmsnorm(x, w_post), layer)

        return x

    # ------------------------------------------------------------------
    # Step 9: Full forward pass
    #   embed -> 30 blocks -> final rmsnorm -> logits (tied embedding)
    # ------------------------------------------------------------------
    def forward(self, token_ids: torch.Tensor,
                cache: list | None = None, position_offset: int = 0) -> torch.Tensor:
        # token_ids: (seq_len,) -> logits: (seq_len, vocab_size)
        # cache: per-request list of 30 layer dicts {"k","v"}, or None (cache-free).
        # position_offset: tokens already in the cache (for RoPE + masking).
        token_ids = token_ids.to(self.device)   # ensure ids are on the model's device
        h = self.embed(token_ids)

        # run every transformer block in order, passing each layer its own cache slot
        for layer in range(self.config.num_layers):
            layer_cache = cache[layer] if cache is not None else None
            h = self.block(h, layer, layer_cache, position_offset)

        # final rmsnorm
        h = self.rmsnorm(h, self.w("model.norm.weight"))

        # project to logits using the TIED embedding table (no separate lm_head)
        logits = h @ self.w("model.embed_tokens.weight").T
        return logits

    def new_cache(self) -> list:
        """Fresh, empty KV cache for one request: 30 layer slots."""
        return [{"k": None, "v": None} for _ in range(self.config.num_layers)]

    # ------------------------------------------------------------------
    # Step 11: Greedy decode loop (autoregressive generation)
    #
    # This is LLM inference. Note the INEFFICIENCY we deliberately keep for
    # now: every step re-runs forward() over the ENTIRE growing sequence and
    # throws away all but the last position's logits. Phase 3 (KV cache) fixes
    # exactly this by reusing past keys/values instead of recomputing them.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_slow(self, token_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Cache-free generation (the wasteful version). Kept for correctness comparison."""
        ids = token_ids
        for _ in range(max_new_tokens):
            logits = self.forward(ids)          # recomputes the WHOLE sequence every step
            next_id = logits[-1].argmax()
            ids = torch.cat([ids, next_id.view(1)])
        return ids

    # ------------------------------------------------------------------
    # Phase 3: cached generation = PREFILL once, then DECODE one token/step.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        cache = self.new_cache()
        ids = token_ids.to(self.device)

        # PREFILL: run the whole prompt once, filling the cache.
        logits = self.forward(ids, cache=cache, position_offset=0)
        next_id = logits[-1].argmax()
        ids: Tensor = torch.cat([ids, next_id.view(1)])

        # DECODE: feed only the ONE new token each step; cache holds the rest.
        for _ in range(max_new_tokens - 1):
            offset = ids.shape[0] - 1                      # tokens already cached
            logits = self.forward(next_id.view(1), cache=cache, position_offset=offset)
            next_id = logits[-1].argmax()
            ids = torch.cat([ids, next_id.view(1)])
        return ids

    # ==================================================================
    # Phase 4: TRUE tensor-batched DECODE
    #
    # Advances N requests by ONE token each, in a SINGLE forward pass.
    # Each request feeds 1 new token but has a DIFFERENT cache length, so we
    # LEFT-PAD every request's cache to the longest and mask the padding.
    #
    # Inputs (parallel lists of length B = num requests):
    #   token_ids : list of int  -- each request's new token to feed
    #   caches    : list of per-request caches (each = list of 30 layer dicts)
    #   offsets   : list of int  -- each request's position_offset (true position
    #               of its new token = number of real tokens already cached)
    # Returns: LongTensor (B,) -- the next token id for each request.
    # ==================================================================
    @torch.no_grad()
    def decode_batch(self, token_ids: list[int], caches: list, offsets: list[int]) -> torch.Tensor:
        cfg = self.config
        B = len(token_ids)
        dev = self.device

        # 1. Stack the B new tokens -> (B,) then embed -> (B, 1, hidden)
        toks = torch.tensor(token_ids, device=dev)          # (B,)
        h = self.embed(toks).unsqueeze(1)                   # (B, 1, hidden)

        # 2. Per-request RoPE cos/sin for its single new token (each at its own
        #    true position = offset). Stack -> (B, 1, head_dim).
        cos_list, sin_list = [], []
        for off in offsets:
            c, s = self.rope_tables(1, off)                 # (1, head_dim) each
            cos_list.append(c)
            sin_list.append(s)
        cos = torch.stack(cos_list)                         # (B, 1, head_dim)
        sin = torch.stack(sin_list)

        # 3. Build a padded-cache-length mask. max_len = longest cache + 1 (the
        #    new token). Each request's valid length = offset + 1.
        max_len = max(offsets) + 1
        valid_len = torch.tensor([off + 1 for off in offsets], device=dev)   # (B,)
        # key position index 0..max_len-1; positions >= valid_len are LEFT-PAD.
        # We LEFT-pad, so real keys occupy the RIGHTMOST valid_len slots.
        key_pos = torch.arange(max_len, device=dev).unsqueeze(0)             # (1, max_len)
        pad_start = max_len - valid_len.unsqueeze(1)                         # (B, 1)
        pad_mask = key_pos < pad_start                                      # (B, max_len) True = PAD

        # 4. Run all 30 blocks in batched form.
        for layer in range(cfg.num_layers):
            h = self._block_batched(h, layer, caches, cos, sin, max_len, pad_mask, offsets)

        # 5. Final norm + logits, take each request's single position.
        h = self.rmsnorm(h, self.w("model.norm.weight"))    # (B, 1, hidden)
        logits = h @ self.w("model.embed_tokens.weight").T  # (B, 1, vocab)
        return logits[:, -1, :].argmax(dim=-1)              # (B,)

    def _block_batched(self, h, layer, caches, cos, sin, max_len, pad_mask, offsets):
        w_in = self.w(f"model.layers.{layer}.input_layernorm.weight")
        w_post = self.w(f"model.layers.{layer}.post_attention_layernorm.weight")
        h = h + self._attn_batched(self.rmsnorm(h, w_in), layer, caches, cos, sin, max_len, pad_mask, offsets)
        h = h + self.mlp(self.rmsnorm(h, w_post), layer)    # mlp is per-token, batches for free
        return h

    def _attn_batched(self, x, layer, caches, cos, sin, max_len, pad_mask, offsets):
        cfg = self.config
        B = x.shape[0]
        dev = self.device
        p = f"model.layers.{layer}.self_attn"
        Wq = self.w(f"{p}.q_proj.weight")
        Wk = self.w(f"{p}.k_proj.weight")
        Wv = self.w(f"{p}.v_proj.weight")
        Wo = self.w(f"{p}.o_proj.weight")

        # project the single new token per request
        q = (x @ Wq.T).view(B, cfg.num_q_heads, cfg.head_dim)    # (B, 9, 64)
        k = (x @ Wk.T).view(B, cfg.num_kv_heads, cfg.head_dim)   # (B, 3, 64)
        v = (x @ Wv.T).view(B, cfg.num_kv_heads, cfg.head_dim)   # (B, 3, 64)

        # RoPE on the new token (cos/sin are (B,1,head_dim) -> squeeze to (B,1,hd))
        # q/k are (B, heads, 64); broadcast cos/sin over heads via (B,1,64).
        cos1 = cos.squeeze(1).unsqueeze(1)   # (B, 1, head_dim)
        sin1 = sin.squeeze(1).unsqueeze(1)
        q = q * cos1 + self.rotate_half(q) * sin1
        k = k * cos1 + self.rotate_half(k) * sin1

        # append each request's new k/v into its own cache, then read padded caches.
        # Build padded (B, kv_heads, max_len, head_dim) K and V.
        Kp = torch.zeros(B, cfg.num_kv_heads, max_len, cfg.head_dim, device=dev)
        Vp = torch.zeros(B, cfg.num_kv_heads, max_len, cfg.head_dim, device=dev)
        for b in range(B):
            lc = caches[b][layer]
            new_k = k[b].unsqueeze(1)   # (kv_heads, 1, head_dim)
            new_v = v[b].unsqueeze(1)
            if lc["k"] is not None:
                fk = torch.cat([lc["k"], new_k], dim=1)   # (kv_heads, valid, head_dim)
                fv = torch.cat([lc["v"], new_v], dim=1)
            else:
                fk, fv = new_k, new_v
            lc["k"], lc["v"] = fk, fv                     # persist grown cache
            valid = fk.shape[1]
            Kp[b, :, max_len - valid:, :] = fk            # LEFT-pad: fill rightmost
            Vp[b, :, max_len - valid:, :] = fv

        # GQA expand kv heads 3 -> 9
        rep = cfg.num_q_heads // cfg.num_kv_heads
        Kp = Kp.repeat_interleave(rep, dim=1)             # (B, 9, max_len, 64)
        Vp = Vp.repeat_interleave(rep, dim=1)

        # attention: q (B,9,64) as (B,9,1,64) vs Kp (B,9,max_len,64)
        q = q.unsqueeze(2)                                # (B, 9, 1, 64)
        scores = q @ Kp.transpose(-2, -1) / math.sqrt(cfg.head_dim)  # (B, 9, 1, max_len)
        # mask padding: pad_mask (B, max_len) -> (B, 1, 1, max_len)
        scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = weights @ Vp                                # (B, 9, 1, 64)
        out = out.squeeze(2).reshape(B, cfg.hidden_size)  # (B, 576)
        out = out @ Wo.T
        return out.unsqueeze(1)                           # (B, 1, 576)


    # ==================================================================
    # Phase 5: PAGED path (dedicated; proves PagedAttention == contiguous).
    #
    # Same math as attention()/forward(), but K/V live in a PagedKVCache
    # (scattered fixed-size blocks) instead of a contiguous torch.cat tensor.
    # ==================================================================
    def _attn_paged(self, x, layer, paged, block_table, positions, position_offset):
        cfg = self.config
        seq_len = x.shape[0]
        p = f"model.layers.{layer}.self_attn"
        Wq = self.w(f"{p}.q_proj.weight")
        Wk = self.w(f"{p}.k_proj.weight")
        Wv = self.w(f"{p}.v_proj.weight")
        Wo = self.w(f"{p}.o_proj.weight")

        q = (x @ Wq.T).view(seq_len, cfg.num_q_heads, cfg.head_dim).transpose(0, 1)   # (9,seq,64)
        k = (x @ Wk.T).view(seq_len, cfg.num_kv_heads, cfg.head_dim).transpose(0, 1)  # (3,seq,64)
        v = (x @ Wv.T).view(seq_len, cfg.num_kv_heads, cfg.head_dim).transpose(0, 1)

        cos, sin = self.rope_tables(seq_len, position_offset)
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)

        # write the new tokens' k/v into the request's blocks, then gather ALL.
        paged.append(layer, block_table, positions, k, v)
        k_all, v_all = paged.gather(layer, block_table)   # (3, total_len, 64)
        total_len = k_all.shape[1]

        rep = cfg.num_q_heads // cfg.num_kv_heads
        k_all = k_all.repeat_interleave(rep, dim=0)       # (9, total_len, 64)
        v_all = v_all.repeat_interleave(rep, dim=0)

        scores = q @ k_all.transpose(-2, -1) / math.sqrt(cfg.head_dim)  # (9, seq, total_len)
        if seq_len > 1:
            q_pos = torch.arange(seq_len, device=self.device).unsqueeze(1) + position_offset
            k_pos = torch.arange(total_len, device=self.device).unsqueeze(0)
            scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v_all                             # (9, seq, 64)
        out = out.transpose(0, 1).reshape(seq_len, cfg.hidden_size)
        return out @ Wo.T

    def _forward_paged(self, token_ids, paged, block_table, positions, position_offset):
        h = self.embed(token_ids)
        for layer in range(self.config.num_layers):
            w_in = self.w(f"model.layers.{layer}.input_layernorm.weight")
            w_post = self.w(f"model.layers.{layer}.post_attention_layernorm.weight")
            h = h + self._attn_paged(self.rmsnorm(h, w_in), layer, paged,
                                     block_table, positions, position_offset)
            h = h + self.mlp(self.rmsnorm(h, w_post), layer)
        h = self.rmsnorm(h, self.w("model.norm.weight"))
        return h @ self.w("model.embed_tokens.weight").T

    @torch.no_grad()
    def generate_paged(self, token_ids, max_new_tokens, paged, block_table):
        """Greedy generation backed by the PAGED KV cache."""
        ids = token_ids.to(self.device)

        # PREFILL: reserve a block-table slot for every prompt token.
        positions = [block_table.append_token() for _ in range(ids.shape[0])]
        logits = self._forward_paged(ids, paged, block_table, positions, position_offset=0)
        next_id = logits[-1].argmax()
        ids = torch.cat([ids, next_id.view(1)])

        # DECODE: one new token per step -> reserve one slot, feed one token.
        for _ in range(max_new_tokens - 1):
            offset = ids.shape[0] - 1
            positions = [block_table.append_token()]
            logits = self._forward_paged(next_id.view(1), paged, block_table, positions, offset)
            next_id = logits[-1].argmax()
            ids = torch.cat([ids, next_id.view(1)])
        return ids
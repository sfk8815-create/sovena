"""Unlimited-OCR MLX Model Implementation.

High-precision OCR model fully implemented in MLX for Apple Silicon acceleration.
Architecture: Vision Encoder (SAM-ViT-B + CLIP-L) → DeepSeek-V2 MoE Language Model.
"""

import math
import time
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import UnlimitedOCRConfig, VisionConfig, LanguageConfig, ProjectorConfig


# =============================================================================
# Utility Functions
# =============================================================================

def _compute_default_rope_freqs(
    dim: int, max_position_embeddings: int = 32768, base: float = 10000.0
) -> mx.array:
    """Compute RoPE frequencies. Returns (max_pos, dim/2) for rotation."""
    theta = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(max_position_embeddings, dtype=mx.float32)
    freqs = mx.outer(t, theta)
    return freqs


def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q, k: [B, heads, seq_len, head_dim]
        cos, sin: [seq_len, half_dim] already sliced/indexed by caller
    """
    B, H, L, D = q.shape
    half_D = D // 2

    # cos/sin are already properly shaped by RotaryEmbedding
    # They should be [L, half_D] or [1, L, half_D]
    if cos.ndim == 3:
        cos = cos.reshape(-1, cos.shape[-1])
        sin = sin.reshape(-1, sin.shape[-1])

    # Ensure correct length
    cos = cos[:L]
    sin = sin[:L]

    # Reshape for broadcasting: [1, 1, L, half_D]
    cos = cos.reshape(1, 1, L, half_D)
    sin = sin.reshape(1, 1, L, half_D)

    def _rotate_half(x):
        x1 = x[..., :half_D]
        x2 = x[..., half_D:]
        return mx.concatenate([-x2, x1], axis=-1)

    # Duplicate cos/sin to full head_dim for element-wise multiply
    cos2 = mx.concatenate([cos, cos], axis=-1)
    sin2 = mx.concatenate([sin, sin], axis=-1)

    q_rot = q * cos2 + _rotate_half(q) * sin2
    k_rot = k * cos2 + _rotate_half(k) * sin2

    return q_rot, k_rot


def silu(x):
    """SiLU activation function."""
    return x * mx.sigmoid(x)


# =============================================================================
# Bicubic (antialias) Resize —— 等价 torch F.interpolate(mode='bicubic',
# antialias=True, align_corners=False)，用于 SAM/CLIP 位置嵌入插值
# =============================================================================

def _cubic_kernel(t: mx.array, a: float) -> mx.array:
    """Keys 型双三次卷积核（偶函数）。

    torch 的 antialias bicubic 移植自 PIL：下采样核参数 a=-0.5（Catmull-Rom）；
    普通（非 antialias）bicubic 用 a=-0.75。经 δ 探测法对 torch 实测验证。
    """
    t = mx.abs(t)
    t2 = t * t
    t3 = t2 * t
    inner = (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0
    outer = a * (t3 - 5.0 * t2 + 8.0 * t - 4.0)
    return mx.where(t < 1.0, inner, mx.where(t < 2.0, outer, mx.zeros_like(outer)))


def _resize_axis_bicubic_aa(x: mx.array, out_len: int) -> mx.array:
    """沿第 0 轴重采样 x [in_len, ...]，PIL/torch-antialias 语义。

    经 one-hot/ramp 探测实证（对齐 torch F.interpolate(mode='bicubic',
    antialias=True, align_corners=False)，上/下采样同一套规则）：
    - 核参数 a=-0.5（Catmull-Rom），上下采样皆同；
    - 下采样 (out < in) 时核宽按 scale 放大（t=(k-center)/scale）；
    - 越界 tap 直接跳过（不折叠边界值），权重在有效 tap 内归一化。
    """
    in_len = x.shape[0]
    if in_len == out_len:
        return x
    scale = in_len / out_len
    antialias = scale > 1.0
    fscale = 1.0 / scale if antialias else 1.0
    support = 2.0 * scale if antialias else 2.0

    j = mx.arange(out_len, dtype=mx.float32)
    center = (j + 0.5) * scale - 0.5
    kmin = mx.ceil(center - support).astype(mx.int32)
    kmax = mx.floor(center + support).astype(mx.int32)

    max_taps = int((kmax - kmin).max().item()) + 1
    taps = mx.arange(max_taps, dtype=mx.int32)
    k = kmin[:, None] + taps[None, :]  # [out, T]
    valid = (k >= 0) & (k <= in_len - 1) & (k <= kmax[:, None])
    k_c = mx.minimum(mx.maximum(k, 0), in_len - 1)
    t = (k.astype(mx.float32) - center[:, None]) * fscale
    w = _cubic_kernel(t, -0.5)
    # 越界 tap 跳过，权重在有效 tap 内归一化
    w = mx.where(valid, w, mx.zeros_like(w))
    w = w / mx.sum(w, axis=1, keepdims=True)
    # out[j, ...] = Σ_t w[j, t] * x[k[j, t], ...]（尾随维度任意）
    rest = x.shape[1:]
    x2 = x.reshape(in_len, -1)
    out = mx.einsum("jt,jtc->jc", w, x2[k_c])
    return out.reshape(out_len, *rest)


def _resize2d_bicubic_aa(grid: mx.array, out_h: int, out_w: int) -> mx.array:
    """grid: [H, W, C] (fp32) → [out_h, out_w, C]。

    torch 的 2D bicubic antialias 是逐轴归一化权重的可分离实现，
    故先沿 H 再沿 W 等价。
    """
    x = _resize_axis_bicubic_aa(grid, out_h)  # [out_h, W, C]
    x = _resize_axis_bicubic_aa(x.transpose(1, 0, 2), out_w).transpose(1, 0, 2)
    return x


# =============================================================================
# RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        # 注意：本移植加载的是原始 PyTorch 权重（w），不是 MLX llama 惯例的 w-1，
        # 因此直接传 self.weight，不能加 1。
        return mx.fast.rms_norm(x, self.weight, self.eps)


# =============================================================================
# RoPE
# =============================================================================

class RotaryEmbedding:
    """Rotary Position Embedding."""

    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self._freqs_cos_sin = None

    def _ensure_freqs(self):
        if self._freqs_cos_sin is None:
            freqs = _compute_default_rope_freqs(self.dim, self.max_position_embeddings, self.base)
            self._freqs_cos_sin = (mx.cos(freqs), mx.sin(freqs))

    @property
    def cos_cached(self):
        self._ensure_freqs()
        return self._freqs_cos_sin[0]

    @property
    def sin_cached(self):
        self._ensure_freqs()
        return self._freqs_cos_sin[1]

    def __call__(self, x, position_ids=None, seq_len=None):
        self._ensure_freqs()
        cos, sin = self.cos_cached, self.sin_cached
        if seq_len is not None:
            cos, sin = cos[:seq_len], sin[:seq_len]
        if position_ids is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]
        return cos, sin


# =============================================================================
# Standard Multi-Head Attention
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention with RoPE."""

    def __init__(self, config: LanguageConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.scale = self.head_dim ** -0.5

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(hidden_states).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(hidden_states).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        cos, sin = self.rotary_emb(q, position_ids=position_ids, seq_len=L)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if past_key_value is not None:
            pk, pv = past_key_value
            k = mx.concatenate([pk, k], axis=2)
            v = mx.concatenate([pv, v], axis=2)

        past_kv = (k, v) if use_cache else None

        # GQA: repeat k/v heads
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        attn_output = attn_weights @ v

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(attn_output)
        return output, past_kv


# =============================================================================
# MLP (SwiGLU)
# =============================================================================

class SwiGLUMLP(nn.Module):
    """SwiGLU MLP used in dense layers and experts."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# MoE (Mixture of Experts)
# =============================================================================

class MoEGate(nn.Module):
    """Top-k gating for MoE."""

    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.norm_topk_prob = config.norm_topk_prob

        # Gate weight: [n_experts, hidden_size]
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))

    def __call__(self, hidden_states: mx.array) -> Tuple[mx.array, mx.array]:
        # hidden_states: [B*L, hidden_size]
        logits = hidden_states.astype(mx.float32) @ self.weight.astype(mx.float32).T

        if self.scoring_func == "softmax":
            scores = mx.softmax(logits, axis=-1)
        else:
            scores = mx.sigmoid(logits)

        # Top-k selection (MLX topk returns indices, then we gather weights)
        topk_indices = mx.argpartition(-scores, kth=self.top_k - 1, axis=-1)[:, :self.top_k]
        # Gather the actual scores for these indices
        topk_weights = mx.take_along_axis(scores, topk_indices, axis=-1)

        if self.norm_topk_prob:
            denom = topk_weights.sum(axis=-1, keepdims=True) + 1e-20
            topk_weights = topk_weights / denom

        return topk_indices, topk_weights


class DeepSeekMoE(nn.Module):
    """DeepSeek-V2 MoE block with shared experts."""

    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.moe_intermediate_size = config.moe_intermediate_size

        # Create routed experts
        self.experts = [
            SwiGLUMLP(config.hidden_size, self.moe_intermediate_size)
            for _ in range(self.n_routed_experts)
        ]

        self.gate = MoEGate(config)

        # Shared experts (2 experts with combined intermediate size)
        if config.n_shared_experts is not None:
            shared_dim = self.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = SwiGLUMLP(config.hidden_size, shared_dim)

    def _moe_infer(self, x: mx.array, topk_ids: mx.array, topk_weights: mx.array) -> mx.array:
        """Inference-time MoE computation."""
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)  # [B*L, D]
        tk_flat = topk_ids.reshape(-1)  # [B*L*K]
        tw_flat = topk_weights.reshape(-1)  # [B*L*K]

        # Count tokens per expert
        import numpy as np
        tk_np = np.array(tk_flat, dtype=np.int32)
        token_counts = np.bincount(tk_np, minlength=self.n_routed_experts)

        # Sort tokens by expert
        sort_indices = mx.argsort(tk_flat)
        repeated_x = mx.repeat(x_flat, self.num_experts_per_tok, axis=0)
        sorted_tokens = repeated_x[sort_indices]
        sorted_weights = tw_flat[sort_indices]

        # Process each expert's tokens
        outputs = []
        start = 0
        for i in range(self.n_routed_experts):
            count = int(token_counts[i])
            if count == 0:
                continue
            end = start + count
            expert_out = self.experts[i](sorted_tokens[start:end].astype(mx.float16))
            expert_out = expert_out * sorted_weights[start:end][:, None]
            outputs.append((sort_indices[start:end], expert_out))
            start = end

        if not outputs:
            return mx.zeros_like(x)

        # Scatter back
        all_indices = mx.concatenate([o[0] for o in outputs], axis=0)
        all_outputs = mx.concatenate([o[1] for o in outputs], axis=0)

        # Restore original order via argsort of indices
        restore = mx.argsort(all_indices)
        final = all_outputs[restore]

        # Sum across top-k experts for each token: (B*L, K, D) → (B*L, D)
        final = final.reshape(B * L, self.num_experts_per_tok, D).sum(axis=1)
        return final.reshape(B, L, D)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        identity = hidden_states
        B, L, D = hidden_states.shape
        x_flat = hidden_states.reshape(-1, D)

        topk_idx, topk_weight = self.gate(x_flat)

        # Reshape routing back
        topk_idx = topk_idx.reshape(B * L, self.num_experts_per_tok)
        topk_weight = topk_weight.reshape(B * L, self.num_experts_per_tok)

        moe_out = self._moe_infer(hidden_states, topk_idx.reshape(B, L, -1), topk_weight.reshape(B, L, -1))

        if hasattr(self, 'shared_experts'):
            moe_out = moe_out + self.shared_experts(identity)

        return moe_out


# =============================================================================
# DeepSeek-V2 Decoder Layer
# =============================================================================

class DeepSeekDecoderLayer(nn.Module):
    """Single decoder layer with attention + MLP/MoE."""

    def __init__(self, config: LanguageConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MultiHeadAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Layer 0 is dense MLP, rest are MoE
        is_dense = layer_idx < config.first_k_dense_replace
        if is_dense:
            self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)
            self.is_moe = False
        else:
            self.mlp = DeepSeekMoE(config)
            self.is_moe = True

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_kv = self.self_attn(
            hidden_states, attention_mask, position_ids, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states

        # MLP / MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_kv


# =============================================================================
# DeepSeek-V2 Language Model
# =============================================================================

class DeepSeekModel(nn.Module):
    """DeepSeek-V2 Language Model (12 layers, MoE)."""

    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepSeekDecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        B, L, _ = inputs_embeds.shape

        # Create causal mask
        if attention_mask is None:
            attention_mask = mx.tril(mx.ones((L, L), dtype=mx.bool_))
            attention_mask = mx.where(attention_mask, 0.0, float('-inf'))
            attention_mask = attention_mask[None, None, :, :]  # [1, 1, L, L]

        # Create position IDs
        if position_ids is None:
            if past_key_values is not None and past_key_values[0] is not None:
                cache_len = past_key_values[0][0].shape[2]
                position_ids = mx.arange(cache_len, cache_len + L, dtype=mx.int32)[None, :]
            else:
                position_ids = mx.arange(0, L, dtype=mx.int32)[None, :]

        hidden_states = inputs_embeds
        new_kv_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values else None
            hidden_states, nkv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=pkv,
                use_cache=use_cache,
            )
            if use_cache:
                new_kv_cache.append(nkv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, new_kv_cache


# =============================================================================
# SAM-ViT-B Vision Encoder
# =============================================================================

class SAMAttention(nn.Module):
    """SAM attention block with decomposed relative position bias.

    官方（deepencoder.py Attention/Block）：每个 block 都使用 decomposed rel pos；
    窗块 (window=14) 的 rel 表长度 27（2*14-1，与权重一致）；
    全局块 (2/5/8/11) 的 rel 表长度 127，运行时线性插值到 2*max(H,W)-1
    （对齐官方 get_rel_pos 的 F.interpolate(mode="linear", align_corners=False)）。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 0,
        use_rel_pos: bool = True,
        input_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.use_rel_pos = use_rel_pos
        if use_rel_pos:
            self.rel_pos_h = mx.zeros((2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = mx.zeros((2 * input_size[1] - 1, self.head_dim))

    def _interp_rel_table(self, R: mx.array, tgt_len: int) -> mx.array:
        """线性插值 rel 表（等价 F.interpolate(mode='linear', align_corners=False)）。"""
        S = R.shape[0]
        if S == tgt_len:
            return R
        src_f = (mx.arange(tgt_len, dtype=mx.float32) + 0.5) * (S / tgt_len) - 0.5
        src_f = mx.maximum(src_f, 0.0)
        i0 = mx.floor(src_f).astype(mx.int32)
        i1 = mx.minimum(i0 + 1, S - 1)
        w = (src_f - i0.astype(mx.float32))[:, None]  # [T, 1]
        return R[i0] * (1 - w) + R[i1] * w

    def _rel_pos_bias(self, q: mx.array, H: int, W: int) -> mx.array:
        """官方 add_decomposed_rel_pos + get_rel_pos 等价实现。

        q: [B, heads, H*W, d]
        返回 additive attn bias: [B, heads, H*W, H*W]
        """
        B, heads, N, d = q.shape
        r_q = q.reshape(B, heads, H, W, d)

        def gather_axis(R: mx.array, L: int) -> mx.array:
            R = self._interp_rel_table(R.astype(mx.float32), 2 * L - 1)  # [2L-1, d]
            idx = mx.arange(L)[:, None] - mx.arange(L)[None, :] + (L - 1)  # [L, L]
            return R[idx]  # [L, L, d]

        Rh = gather_axis(self.rel_pos_h, H)  # [H, kh, d]
        Rw = gather_axis(self.rel_pos_w, W)  # [W, kw, d]

        # rel_h[b,h,i,j,kh] = Σ_c r_q[b,h,i,j,c]·Rh[i,kh,c]
        rel_h = mx.einsum("bhijc,ikc->bhijk", r_q, Rh)  # [B, heads, H, W, H]
        rel_w = mx.einsum("bhijc,jkc->bhijk", r_q, Rw)  # [B, heads, H, W, W]
        bias = rel_h[..., None] + rel_w[..., None, :]
        return bias.reshape(B, heads, N, H * W).astype(q.dtype)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        # 官方 Block.forward：先对 norm 后的 x 零填充到窗口整数倍，再进 Attention
        # （padded 位置的 qkv = 线性层 bias，非零！），因此必须在 qkv 之前填充。
        pad_h = pad_w = 0
        Hp, Wp = H, W
        if self.window_size > 0:
            ws = self.window_size
            pad_h = (ws - H % ws) % ws
            pad_w = (ws - W % ws) % ws
            if pad_h > 0 or pad_w > 0:
                x = mx.pad(
                    x.reshape(B, H, W, C),
                    [(0, 0), (0, pad_h), (0, pad_w), (0, 0)],
                ).reshape(B, (H + pad_h) * (W + pad_w), C)
                Hp, Wp = H + pad_h, W + pad_w

        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(0, 2, 1, 3)  # [B, heads, Np, head_dim]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Window attention
        if self.window_size > 0:
            out = self._window_attention(q, k, v, Hp, Wp)
        else:
            attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            if self.use_rel_pos:
                attn = attn + self._rel_pos_bias(q, H, W)
            attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
            out = attn @ v

        out = out.transpose(0, 2, 1, 3).reshape(B, Hp * Wp, C)

        # 裁掉 padding（等价官方 window_unpartition 的 :H, :W 裁剪）
        if pad_h > 0 or pad_w > 0:
            out = out.reshape(B, Hp, Wp, C)[:, :H, :W, :].reshape(B, H * W, C)
        return self.proj(out)

    def _window_attention(self, q, k, v, Hp, Wp):
        """窗注意力（x 已在 __call__ 中填充到窗口整数倍）。

        rel_pos 使用窗块 27 长表（q_size=k_size=14，无需插值）。
        """
        B, heads, N, d = q.shape
        ws = self.window_size
        nw_h, nw_w = Hp // ws, Wp // ws

        def window_partition(x):
            # x: [B, heads, Hp*Wp, d]
            x = x.reshape(B, heads, nw_h, ws, nw_w, ws, d)
            x = x.transpose(0, 1, 2, 4, 3, 5, 6)  # [B, heads, nw_h, nw_w, ws, ws, d]
            return x.reshape(B * nw_h * nw_w, heads, ws * ws, d)

        def window_reverse(x):
            x = x.reshape(B, heads, nw_h, nw_w, ws, ws, d)
            x = x.transpose(0, 1, 2, 4, 3, 5, 6)  # [B, heads, nw_h, ws, nw_w, ws, d]
            return x.reshape(B, heads, Hp * Wp, d)

        q_w = window_partition(q)
        k_w = window_partition(k)
        v_w = window_partition(v)

        attn = (q_w @ k_w.transpose(0, 1, 3, 2)) * self.scale
        if self.use_rel_pos:
            # 窗内 q_size=k_size=ws → rel 表 27 = 2*ws-1，直接使用
            attn = attn + self._rel_pos_bias(q_w, ws, ws)
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
        out_w = attn @ v_w

        return window_reverse(out_w)


class SAMMLP(nn.Module):
    """SAM MLP block."""

    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, dim)

    def __call__(self, x):
        return self.lin2(nn.gelu(self.lin1(x)))


class SAMBlock(nn.Module):
    """SAM ViT block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        window_size: int = 0,
        use_rel_pos: bool = True,
        input_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SAMAttention(
            dim, num_heads,
            window_size=window_size,
            use_rel_pos=use_rel_pos,
            input_size=input_size,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SAMMLP(dim, int(dim * mlp_ratio))

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Patch embedding for SAM. Uses NHWC format for MLX."""

    def __init__(self, kernel_size=16, stride=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size, stride=stride, bias=True)

    def __call__(self, x):
        # x: [B, H, W, C] (NHWC)
        return self.proj(x)


class SAMVisionEncoder(nn.Module):
    """SAM-ViT-B vision encoder."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.img_size = config.sam_img_size
        self.patch_size = config.sam_patch_size
        grid_size = self.img_size // self.patch_size  # 64

        self.patch_embed = PatchEmbed(
            kernel_size=config.sam_patch_size,
            stride=config.sam_patch_size,
            in_chans=3,
            embed_dim=config.sam_embed_dim,
        )
        self.pos_embed = mx.zeros((1, grid_size, grid_size, config.sam_embed_dim))

        input_size = (grid_size, grid_size)
        self.blocks = []
        for i in range(config.sam_depth):
            use_global = i in config.sam_global_attn_indexes
            window_size = 0 if use_global else config.sam_window_size
            # 官方 Block：input_size if window_size == 0 else (window_size, window_size)
            # → 窗块 rel 表 27 (2*14-1)，全局块 127 (2*64-1)，与权重一致
            attn_input_size = (
                input_size if use_global
                else (config.sam_window_size, config.sam_window_size)
            )
            self.blocks.append(SAMBlock(
                dim=config.sam_embed_dim,
                num_heads=config.sam_num_heads,
                mlp_ratio=config.sam_mlp_ratio,
                window_size=window_size,
                input_size=attn_input_size,
            ))

        # Neck
        self.neck = nn.Sequential(
            nn.Conv2d(config.sam_embed_dim, config.sam_out_chans, 1, bias=False),
            nn.LayerNorm(config.sam_out_chans, eps=1e-6),
            nn.Conv2d(config.sam_out_chans, config.sam_out_chans, 3, padding=1, bias=False),
            nn.LayerNorm(config.sam_out_chans, eps=1e-6),
        )

        # Downsampling convolutions
        self.net_2 = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, H, W, C] (NHWC format for MLX)
        B, H_in, W_in, C_in = x.shape

        x = self.patch_embed(x)  # [B, H_p, W_p, 768]
        H_p, W_p = x.shape[1], x.shape[2]

        # 位置嵌入：官方 get_abs_pos_sam —— bicubic + antialias 插值
        if self.pos_embed.shape[1] != H_p or self.pos_embed.shape[2] != W_p:
            pos = _resize2d_bicubic_aa(
                self.pos_embed[0].astype(mx.float32), H_p, W_p
            )[None].astype(x.dtype)
        else:
            pos = self.pos_embed

        # Add positional embedding (flatten to sequence)
        x = x.reshape(B, H_p * W_p, -1)  # [B, N, 768]
        pos = pos.reshape(1, H_p * W_p, -1)
        x = x + pos

        for blk in self.blocks:
            x = blk(x)

        # Back to NHWC for convolution
        x = x.reshape(B, H_p, W_p, -1)  # [B, 64, 64, 768]

        # Neck (Conv2d with NHWC)
        x = self.neck(x)  # [B, 64, 64, 256]

        # Downsampling (NHWC)
        x = self.net_2(x)  # [B, 32, 32, 512]
        x = self.net_3(x)  # [B, 16, 16, 1024]

        # Return in NHWC then convert to NCHW for CLIP compatibility
        return x


# =============================================================================
# CLIP-L Vision Encoder
# =============================================================================

class CLIPAttention(nn.Module):
    """CLIP multi-head self-attention."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0].transpose(0, 2, 1, 3), qkv[:, :, 1].transpose(0, 2, 1, 3), qkv[:, :, 2].transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    """CLIP MLP with QuickGELU."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=True)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=True)

    def __call__(self, x):
        # QuickGELU: fc1 → QuickGELU → fc2
        h = self.fc1(x)
        h = h * mx.sigmoid(1.702 * h)
        return self.fc2(h)


class CLIPTransformerLayer(nn.Module):
    """CLIP transformer layer."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = CLIPMLP(hidden_size, ffn_hidden_size)

    def __call__(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPVisionEmbeddings(nn.Module):
    """CLIP vision embeddings that takes SAM features as input."""

    def __init__(self, hidden_size: int = 1024, image_size: int = 224, patch_size: int = 14):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.class_embedding = mx.zeros((hidden_size,))

        # Patch embedding (projects SAM features) - NHWC conv
        self.patch_embedding = nn.Conv2d(3, hidden_size, patch_size, stride=patch_size, bias=False)

        # Position embedding
        self.position_embedding = nn.Embedding(self.num_positions, hidden_size)
        self.position_ids = mx.arange(self.num_positions)[None, :]

    def __call__(self, pixel_values, patch_embeds=None):
        batch_size = pixel_values.shape[0]

        if patch_embeds is not None:
            # Use pre-computed SAM features
            # patch_embeds: [B, H, W, C] (NHWC from SAM)
            B, H, W, C = patch_embeds.shape
            patch_embeds = patch_embeds.reshape(B, H * W, C)
        else:
            # Use raw conv on NHWC input
            patch_embeds = self.patch_embedding(pixel_values)
            B, H, W, C = patch_embeds.shape
            patch_embeds = patch_embeds.reshape(B, H * W, C)

        class_embeds = mx.tile(self.class_embedding.reshape(1, 1, -1), (batch_size, 1, 1))
        embeddings = mx.concatenate([class_embeds, patch_embeds], axis=1)

        # 位置嵌入：官方 get_abs_pos —— cls 位置保留，patch 网格
        # bicubic + antialias 插值到当前网格（局部 10x10 / 全局 16x16）
        pos = self.position_embedding.weight.astype(mx.float32)  # [257, C]
        cls_pos = pos[:1]
        src_grid = int(math.sqrt(pos.shape[0] - 1))
        tgt_grid = int(math.sqrt(embeddings.shape[1] - 1))
        if tgt_grid != src_grid:
            grid = _resize2d_bicubic_aa(
                pos[1:].reshape(src_grid, src_grid, -1), tgt_grid, tgt_grid
            )
            new_pos = mx.concatenate(
                [cls_pos, grid.reshape(tgt_grid * tgt_grid, -1)], axis=0
            )
        else:
            new_pos = pos
        embeddings = embeddings + new_pos[None].astype(embeddings.dtype)

        return embeddings


class CLIPVisionTransformer(nn.Module):
    """CLIP-L vision transformer."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(
            hidden_size=config.clip_hidden_size,
            image_size=config.clip_image_size,
            patch_size=config.clip_patch_size,
        )
        self.pre_layrnorm = nn.LayerNorm(config.clip_hidden_size, eps=config.clip_layernorm_epsilon)
        self.transformer = nn.Sequential(*[
            CLIPTransformerLayer(
                config.clip_hidden_size,
                config.clip_num_heads,
                config.clip_ffn_hidden_size,
                eps=config.clip_layernorm_epsilon,
            )
            for _ in range(config.clip_num_layers)
        ])

    def __call__(self, pixel_values, patch_embeds=None):
        x = self.embeddings(pixel_values, patch_embeds)
        x = self.pre_layrnorm(x)
        x = self.transformer(x)
        return x


# =============================================================================
# Projector
# =============================================================================

class MlpProjector(nn.Module):
    """Linear projector from vision to language space."""

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.layers = nn.Linear(config.input_dim, config.n_embed, bias=True)

    def __call__(self, x):
        return self.layers(x)


# =============================================================================
# Unlimited OCR Model
# =============================================================================

@dataclass
class ModelOutput:
    logits: mx.array
    past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None


class UnlimitedOCRModel(nn.Module):
    """Complete Unlimited-OCR model with vision + language.

    Architecture:
    Image → SAM-ViT-B → CLIP-L → Projector → DeepSeek-V2 MoE → Text
    """

    def __init__(self, config: UnlimitedOCRConfig):
        super().__init__()
        self.config = config

        # Vision
        self.sam_model = SAMVisionEncoder(config.vision)
        self.vision_model = CLIPVisionTransformer(config.vision)

        # Projector: 2048 → 1280
        self.projector = MlpProjector(config.projector)

        # Language
        self.language_model = DeepSeekModel(config.language)
        self.lm_head = nn.Linear(config.language.hidden_size, config.language.vocab_size, bias=False)

        # Image special tokens
        embed_std = 1.0 / math.sqrt(config.language.hidden_size)
        self.image_newline = mx.random.normal((config.language.hidden_size,)) * embed_std
        self.view_seperator = mx.random.normal((config.language.hidden_size,)) * embed_std

    def encode_images(self, images: mx.array, images_spatial_crop=None) -> List[mx.array]:
        """Encode images through vision encoder.

        Args:
            images: List of [patches, original] image tensors (in NCHW from preprocessing)
            images_spatial_crop: List of (width_crops, height_crops) tuples

        Returns:
            List of image feature tensors [N, hidden_size]
        """
        all_features = []

        for idx, image_pair in enumerate(images):
            patches = image_pair[0]  # [N, 3, 640, 640] NCHW
            image_ori = image_pair[1]  # [1, 3, 1024, 1024] NCHW

            has_patches = patches is not None and patches.shape[0] > 0

            # Convert to NHWC for MLX conv
            def to_nhwc(t):
                if t is None:
                    return None
                ndim = len(t.shape)
                if ndim == 4:
                    return t.transpose(0, 2, 3, 1)  # NCHW → NHWC
                return t

            patches_nhwc = to_nhwc(patches)
            image_ori_nhwc = to_nhwc(image_ori)

            if has_patches and images_spatial_crop is not None:
                crop_shape = images_spatial_crop[idx]
                width_crop_num, height_crop_num = crop_shape

                # Process patches (local features)
                sam_local = self.sam_model(patches_nhwc)  # [P, 16, 16, 1024]
                clip_local = self.vision_model(patches_nhwc, sam_local)  # [P, 257, 1024]

                # Combine: CLIP[:, 1:] + SAM flatten
                # SAM: [P, 16, 16, 1024] → [P, 256, 1024]
                sam_flat = sam_local.reshape(patches.shape[0], -1, 1024)
                local_feats = mx.concatenate([
                    clip_local[:, 1:, :],  # [P, 256, 1024]
                    sam_flat,  # [P, 256, 1024]
                ], axis=-1)  # [P, 256, 2048]
                local_feats = self.projector(local_feats)  # [P, 256, 1280]

                # Process original (global features)
                sam_global = self.sam_model(image_ori_nhwc)  # [1, 16, 16, 1024]
                clip_global = self.vision_model(image_ori_nhwc, sam_global)  # [1, 257, 1024]

                sam_gflat = sam_global.reshape(1, -1, 1024)
                global_feats = mx.concatenate([
                    clip_global[:, 1:, :],  # [1, 256, 1024]
                    sam_gflat,  # [1, 256, 1024]
                ], axis=-1)  # [1, 256, 2048]
                global_feats = self.projector(global_feats)  # [1, 256, 1280]

                # Reshape and organize
                _, hw_g, nd = global_feats.shape
                h_g = w_g = int(hw_g ** 0.5)

                _, hw_l, nd2 = local_feats.shape
                h_l = w_l = int(hw_l ** 0.5)

                # Global: reshape to 2D and add newlines
                gf = global_feats.reshape(h_g, w_g, nd)
                gf = mx.concatenate([gf, mx.tile(self.image_newline[None, None, :], (h_g, 1, 1))], axis=1)
                gf = gf.reshape(-1, nd)

                # Local: reshape grid
                lf = local_feats.reshape(height_crop_num, width_crop_num, h_l, w_l, nd2)
                lf = lf.transpose(0, 2, 1, 3, 4).reshape(height_crop_num * h_l, width_crop_num * w_l, nd2)
                lf = mx.concatenate([lf, mx.tile(self.image_newline[None, None, :], (height_crop_num * h_l, 1, 1))], axis=1)
                lf = lf.reshape(-1, nd2)

                # Concat: local + global + separator
                full_feats = mx.concatenate([lf, gf, self.view_seperator[None, :]], axis=0)
                all_features.append(full_feats)

            else:
                # Multiple images or single image without crop
                if len(image_ori_nhwc.shape) == 3:
                    image_ori_nhwc = image_ori_nhwc[None, :, :, :]

                num_imgs = image_ori_nhwc.shape[0]
                for i in range(num_imgs):
                    img = image_ori_nhwc[i:i+1]
                    sam_out = self.sam_model(img)
                    clip_out = self.vision_model(img, sam_out)

                    sam_flat = sam_out.reshape(1, -1, 1024)
                    gf = mx.concatenate([
                        clip_out[:, 1:, :],
                        sam_flat,
                    ], axis=-1)
                    gf = self.projector(gf)

                    _, hw, nd = gf.shape
                    h = w = int(hw ** 0.5)

                    gf_2d = gf.reshape(h, w, nd)
                    gf_2d = mx.concatenate([gf_2d, mx.tile(self.image_newline[None, None, :], (h, 1, 1))], axis=1)
                    gf_2d = gf_2d.reshape(-1, nd)

                    full_feats = mx.concatenate([gf_2d, self.view_seperator[None, :]], axis=0)
                    all_features.append(full_feats)

        return all_features

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None,
        inputs_embeds: Optional[mx.array] = None,
        images: Optional[List[mx.array]] = None,
        images_seq_mask: Optional[mx.array] = None,
        images_spatial_crop: Optional[List[Tuple[int, int]]] = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        # Inject image features into embeddings
        if images is not None and images_seq_mask is not None:
            image_features = self.encode_images(images, images_spatial_crop)

            for idx, img_feats in enumerate(image_features):
                if img_feats is not None and img_feats.shape[0] > 0:
                    mask_flat = images_seq_mask[idx].reshape(-1)
                    # 特征对应 mask 为 True 的连续区间（mlx 0.32 无 .at[].set()，按区间拼接）
                    start = int(mx.argmax(mask_flat).item())
                    n = img_feats.shape[0]
                    row = inputs_embeds[idx]
                    body = mx.where(
                        mask_flat[start:start + n].reshape(-1, 1),
                        img_feats, row[start:start + n],
                    )
                    new_row = mx.concatenate(
                        [row[:start], body, row[start + n:]], axis=0
                    )
                    inputs_embeds = mx.concatenate(
                        [inputs_embeds[:idx], new_row[None], inputs_embeds[idx + 1:]],
                        axis=0,
                    )

        hidden_states, new_kv = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)
        return ModelOutput(logits=logits, past_key_values=new_kv)

    def generate(
        self,
        input_ids: mx.array,
        images: Optional[List] = None,
        images_seq_mask: Optional[mx.array] = None,
        images_spatial_crop: Optional[List] = None,
        max_length: int = 32768,
        temperature: float = 0.0,
        eos_token_id: int = 1,
    ) -> mx.array:
        """Autoregressive text generation."""
        generated = [input_ids]
        past_kv = None
        use_images = (images is not None)
        gen_start = time.time()

        for step in range(max_length):
            if step == 0:
                # Prefill: process full sequence with images
                output = self(
                    input_ids=input_ids,
                    images=images if use_images else None,
                    images_seq_mask=images_seq_mask if use_images else None,
                    images_spatial_crop=images_spatial_crop if use_images else None,
                    use_cache=True,
                )
            else:
                # Decode: process only the last token
                output = self(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_kv,
                    use_cache=True,
                )

            past_kv = output.past_key_values
            logits = output.logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature
                probs = mx.softmax(logits.astype(mx.float32), axis=-1)
                next_token = mx.random.categorical(probs, axis=-1).reshape(1, 1)
            else:
                next_token = mx.argmax(logits, axis=-1, keepdims=True)

            generated.append(next_token)
            input_ids = next_token

            if step % 50 == 0 and step > 0:
                el = time.time() - gen_start
                print(f"  [gen] step={step} {step/el:.1f} t/s", flush=True)

            if next_token.item() == eos_token_id:
                break

        return mx.concatenate(generated, axis=1)

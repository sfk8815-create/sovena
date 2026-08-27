"""P0-4c 诊断：PyTorch fp32 参考实现 vs MLX 移植，逐层对比 LM。

权重直接读 LoJexLLM 的 model.safetensors（language_model.* 键），
参考实现按官方 modeling_deepseekv2.py（use_mla=False → LlamaAttention + MoEGate）。
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import safetensors.torch as st

MODEL_DIR = "/Users/sfk-studio/.lmstudio/models/LoJexLLM/Unlimited-OCR-MLX"
W = {k: v.float() for k, v in st.load_file(os.path.join(MODEL_DIR, "model.safetensors"), device="cpu").items()}

P = "language_model"
H, NH, HD, EPS = 1280, 10, 128, 1e-6
N_E, N_S, TOPK, EID = 64, 2, 6, 896
DENSE_I = 6848

def rms(x, w):
    v = x.pow(2).mean(-1, keepdim=True)
    return w * (x * torch.rsqrt(v + EPS))

def rope_cs(pos, dim=HD, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(pos, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def silu(x):
    return x * torch.sigmoid(x)

def swiglu(x, g, u, d):
    return (silu(x @ g.T) * (x @ u.T)) @ d.T

def attn(x, i):
    ln = rms(x, W[f"{P}.layers.{i}.input_layernorm.weight"])
    B, L, _ = ln.shape
    q = (ln @ W[f"{P}.layers.{i}.self_attn.q_proj.weight"].T).view(B, L, NH, HD).transpose(1, 2)
    k = (ln @ W[f"{P}.layers.{i}.self_attn.k_proj.weight"].T).view(B, L, NH, HD).transpose(1, 2)
    v = (ln @ W[f"{P}.layers.{i}.self_attn.v_proj.weight"].T).view(B, L, NH, HD).transpose(1, 2)
    cos, sin = rope_cs(L)
    cos, sin = cos[None, None], sin[None, None]
    q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
    scores = (q @ k.transpose(-1, -2)) * (HD ** -0.5)
    mask = torch.tril(torch.ones(L, L)).bool()
    scores = scores.masked_fill(~mask[None, None], float("-inf"))
    out = torch.softmax(scores, dim=-1) @ v
    out = out.transpose(1, 2).reshape(B, L, NH * HD)
    return out @ W[f"{P}.layers.{i}.self_attn.o_proj.weight"].T

def moe(x, i):
    B, L, _ = x.shape
    xf = x.reshape(-1, H)
    logits = xf @ W[f"{P}.layers.{i}.mlp.gate.weight"].T
    scores = logits.softmax(dim=-1)
    topk_w, topk_idx = torch.topk(scores, TOPK, dim=-1, sorted=False)
    y = torch.zeros_like(xf)
    for t in range(xf.shape[0]):
        for j in range(TOPK):
            e = topk_idx[t, j].item()
            g = W[f"{P}.layers.{i}.mlp.experts.{e}.gate_proj.weight"]
            u = W[f"{P}.layers.{i}.mlp.experts.{e}.up_proj.weight"]
            d = W[f"{P}.layers.{i}.mlp.experts.{e}.down_proj.weight"]
            y[t] += swiglu(xf[t:t+1], g, u, d).squeeze(0) * topk_w[t, j]
    y = y + swiglu(xf,
                   W[f"{P}.layers.{i}.mlp.shared_experts.gate_proj.weight"],
                   W[f"{P}.layers.{i}.mlp.shared_experts.up_proj.weight"],
                   W[f"{P}.layers.{i}.mlp.shared_experts.down_proj.weight"])
    return y.reshape(B, L, H)

def ref_forward(ids):
    h = W[f"{P}.embed_tokens.weight"][torch.tensor(ids)][None]
    outs = []
    for i in range(12):
        x = h + attn(h, i)
        if i == 0:
            m = swiglu(rms(x, W[f"{P}.layers.0.post_attention_layernorm.weight"]).reshape(-1, H),
                       W[f"{P}.layers.0.mlp.gate_proj.weight"],
                       W[f"{P}.layers.0.mlp.up_proj.weight"],
                       W[f"{P}.layers.0.mlp.down_proj.weight"]).reshape(1, -1, H)
        else:
            m = moe(rms(x, W[f"{P}.layers.{i}.post_attention_layernorm.weight"]), i)
        h = x + m
        outs.append(h[0].clone())
    h = rms(h, W[f"{P}.norm.weight"])
    logits = h[0] @ W["lm_head.weight"].T
    return outs, logits

# ── 输入 ──
from ocr_port.inference import load_tokenizer
tok = load_tokenizer(MODEL_DIR)
text = "The capital of France is"
ids = [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)
print("ids:", ids)

outs, logits = ref_forward(ids)
print("\n== PyTorch fp32 参考 ==")
top = torch.topk(logits[-1], 5).indices.tolist()
print("top5 next tokens:", [(t, repr(tok.decode([t]))) for t in top])
# 贪心生成 20 步
gen = list(ids)
with torch.no_grad():
    for _ in range(20):
        _, lg = ref_forward(gen)
        nt = int(torch.argmax(lg[-1]))
        gen.append(nt)
        if nt == 1:
            break
print("贪心续写:", repr(tok.decode(gen[len(ids):], skip_special_tokens=True)))

# ── MLX 对比 ──
print("\n== MLX fp16 移植 逐层对比 ==")
import mlx.core as mx
from ocr_port.inference import load_model, create_attention_mask
model = load_model(MODEL_DIR)

import numpy as np
def cos(a_torch, a_mlx):
    a = a_torch.detach().numpy().astype(np.float32).reshape(-1)
    b = np.array(a_mlx, dtype=np.float32).reshape(-1)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

h_m = model.language_model.embed_tokens(mx.array([ids], dtype=mx.int32))
print(f"embed     cos={cos(W[f'{P}.embed_tokens.weight'][torch.tensor(ids)], h_m[0]):.6f}")
L = len(ids)
mask = create_attention_mask(L)
pos = mx.arange(0, L, dtype=mx.int32)[None, :]
for i in range(12):
    h_m, _ = model.language_model.layers[i](h_m, mask, pos, None, False)
    print(f"layer {i:2d}  cos={cos(outs[i], h_m[0]):.6f}")
h_m = model.language_model.norm(h_m)
lg_m = model.lm_head(h_m)[0, -1]
lg_t = logits[-1]
top_m = np.argsort(np.array(lg_m))[-5:][::-1].tolist()
print("MLX top5:", [(int(t), repr(tok.decode([int(t)]))) for t in top_m])
print("ref  top5:", [(t, repr(tok.decode([t]))) for t in top])

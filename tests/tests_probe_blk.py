"""block 0 内部逐组件对比：norm1 → qkv → window attn → proj → norm2 → mlp。"""
import sys, os
sys.path.insert(0, "/tmp/dsocr_ref")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx
import safetensors.torch
import deepencoder

MODEL_DIR = "/Users/sfk-studio/.lmstudio/models/LoJexLLM/Unlimited-OCR-MLX"
W = safetensors.torch.load_file(os.path.join(MODEL_DIR, "model.safetensors"))

sam = deepencoder.build_sam_vit_b().float().eval()
sam.load_state_dict({k[len("sam_model."):]: v.float() for k, v in W.items() if k.startswith("sam_model.")}, strict=True)

from ocr_port.inference import load_model as mlx_load
msam = mlx_load(MODEL_DIR, dtype="float32").sam_model

rng = np.random.default_rng(7)
B = 2
x = rng.standard_normal((B, 40, 40, 768)).astype(np.float32)

def rep(name, ref, out):
    ref = ref.detach().numpy() if torch.is_tensor(ref) else np.asarray(ref)
    out = np.asarray(out)
    if out.size == ref.size and out.shape != ref.shape:
        out = out.reshape(ref.shape)
    print(f"  {name}: ref{ref.shape} out{tuple(out.shape)} max|diff|={np.abs(ref-out).max():.3e}")

tb, mb = sam.blocks[0], msam.blocks[0]

with torch.no_grad():
    t_n1 = tb.norm1(torch.from_numpy(x))
m_n1 = mb.norm1(mx.array(x))
rep("norm1", t_n1, m_n1)

# qkv
with torch.no_grad():
    t_qkv = tb.attn.qkv(t_n1)  # [B,40,40,2304]
m_qkv = mb.attn.qkv(mx.array(np.array(m_n1)))
rep("qkv", t_qkv, m_qkv)

# 整个 attention（MLX 侧需展平 [B,N,C]）
with torch.no_grad():
    t_a = tb.attn(t_n1)
m_a = mb.attn(mx.array(np.array(m_n1)).reshape(B, 40 * 40, 768))
rep("attn_out", t_a, m_a)

# mlp
with torch.no_grad():
    t_m = tb.mlp(tb.norm2(t_x2 := t_n1))
m_m = mb.mlp(mb.norm2(mx.array(np.array(m_n1)).reshape(B, 40 * 40, 768)))
rep("mlp", t_m, m_m)

# 完整 block
with torch.no_grad():
    t_b = tb(torch.from_numpy(x))
m_b = mb(mx.array(x).reshape(B, 40 * 40, 768))
rep("block0", t_b, m_b)

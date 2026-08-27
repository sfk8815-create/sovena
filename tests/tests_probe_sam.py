"""逐阶段对比 SAM：patch_embed / pos_embed / 每 block / neck / net_2 / net_3。"""
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
sam_sd = {k[len("sam_model."):]: v.float() for k, v in W.items() if k.startswith("sam_model.")}
sam.load_state_dict(sam_sd, strict=True)

from ocr_port.inference import load_model as mlx_load
mlx_model = mlx_load(MODEL_DIR, dtype="float32")
msam = mlx_model.sam_model

# 输入：640 crop → 40x40（下采样 pos_embed 64→40）
rng = np.random.default_rng(7)
x = rng.standard_normal((2, 3, 640, 640)).astype(np.float32)
x_nhwc = x.transpose(0, 2, 3, 1)

def rep(name, ref, out):
    # ref: torch [B,C,H,W] 或 [B,N,C]；out: mlx
    ref = ref.detach().numpy() if torch.is_tensor(ref) else ref
    out = np.array(out)
    if ref.ndim == 4 and out.ndim == 4 and ref.shape != out.shape:
        ref = ref.transpose(0, 2, 3, 1)
    if ref.ndim == 3 and out.ndim == 3 and ref.shape[1] != out.shape[1] and ref.shape[1] == out.shape[2]:
        ref = ref.transpose(0, 2, 1)
    d = np.abs(ref - out).max()
    print(f"  {name}: ref{ref.shape} out{out.shape} max|diff|={d:.3e}")

with torch.no_grad():
    # 1. patch_embed
    t_pe = sam.patch_embed(torch.from_numpy(x))  # [B,768,40,40]
    m_pe = msam.patch_embed(mx.array(x_nhwc))     # [B,40,40,768]
    rep("patch_embed", t_pe, m_pe)

    # 2. pos_embed (插值 64→40)
    t_pos = deepencoder.get_abs_pos_sam(sam.pos_embed, t_pe.size(2))  # [1,T,T,C]? 返回 permute 后
    t_pos = t_pos.permute(0, 2, 3, 1) if t_pos.dim() == 4 and t_pos.shape[1] == 768 else t_pos
    pos0 = msam.pos_embed[0].astype(mx.float32)
    m_pos = mx.zeros((40, 40, 768))
    from ocr_port.model import _resize2d_bicubic_aa
    m_pos = _resize2d_bicubic_aa(pos0, 40, 40)
    t_p = sam.pos_embed.detach().permute(0, 3, 1, 2).float()
    t_p = F.interpolate(t_p, size=(40, 40), mode="bicubic", antialias=True, align_corners=False).permute(0, 2, 3, 1)[0].numpy()
    rep("pos_embed_interp", t_p, m_pos)

    # 3. block 逐个（PatchEmbed 已返回 NHWC [B,40,40,768]）
    t_x = t_pe + t_p[None]  # [B,40,40,768]
    m_x = mx.array(np.array(m_pe)) + m_pos[None]
    m_seq = m_x.reshape(2, 40 * 40, -1)
    for i, (tb, mb) in enumerate(zip(sam.blocks, msam.blocks)):
        with torch.no_grad():
            t_x = tb(t_x)
        m_seq = mb(m_seq)
        if i in (0, 1, 2, 5, 11):
            m_g = np.array(m_seq).reshape(2, 40, 40, 768)
            rep(f"block[{i}]", t_x, m_g)
    # block 输出 NHWC [B,40,40,768]
    t_final = t_x
    m_final = np.array(m_seq).reshape(2, 40, 40, 768)
    rep("blocks_final", t_final, m_final)

    # 4. neck
    t_neck = sam.neck(t_final.permute(0, 3, 1, 2))
    m_neck = msam.neck(mx.array(m_final))
    rep("neck", t_neck, m_neck)

    t2 = sam.net_2(t_neck)
    m2 = msam.net_2(mx.array(np.array(m_neck)))
    rep("net_2", t2, m2)

    t3 = sam.net_3(t2.clone())
    m3 = msam.net_3(mx.array(np.array(m2)))
    rep("net_3", t3, m3)

print("done")

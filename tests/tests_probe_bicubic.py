"""探针：对比 MLX 上采样 bicubic 与 torch 单输出像素的 tap/权重。"""
import sys, os
sys.path.insert(0, "/Users/sfk-studio/Desktop/文献流设计/litflow")
import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx
from ocr_port.model import _resize_axis_bicubic_aa, _cubic_kernel

rng = np.random.default_rng(42)
in_len, out_len = 8, 10
x = rng.standard_normal((in_len, 1)).astype(np.float32)  # [L, 1]

# torch 参考（1D → 用 2D [1,1,L,1]，只看 axis）
ref = F.interpolate(torch.from_numpy(x)[None, None], size=(out_len, 1), mode="bicubic", align_corners=False)[0, 0].numpy()

out = np.array(_resize_axis_bicubic_aa(mx.array(x), out_len))
print("max|diff| =", np.abs(ref - out).max())

# 手工检查 j=0
scale = in_len / out_len
center = 0.5 * scale - 0.5
kmin = int(np.ceil(center - 2.0)); kmax = int(np.floor(center + 2.0))
print(f"j=0: center={center}, kmin={kmin}, kmax={kmax}")
for k in range(kmin, kmax + 1):
    k_c = min(max(k, 0), in_len - 1)
    w = float(_cubic_kernel(mx.array(abs(k - center)), -0.75))
    print(f"  k={k:3d} k_c={k_c} t={k - center:+.3f} w={w:+.6f} x[k_c]={x[k_c,0]:+.4f}")
manual = sum(float(_cubic_kernel(mx.array(abs(k - center)), -0.75)) * x[min(max(k,0),in_len-1),0] for k in range(kmin, kmax+1))
print(f"manual out[0] = {manual:.6f}, ours={out[0,0]:.6f}, torch={ref[0,0]:.6f}")

# torch 的 tap 重建：floor(center), tx
import math
f = math.floor(center); tx = center - f
print(f"torch: floor={f}, tx={tx:.3f}")

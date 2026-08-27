"""P0-4d：视觉塔 MLX vs 官方 PyTorch (fp32) 数值对齐验证。

Part 1: bicubic+antialias 插值 vs torch F.interpolate
Part 2: SAM/CLIP/Projector/最终视觉特征 逐阶段对比
"""
import sys, os
import numpy as np

sys.path.insert(0, "/tmp/dsocr_ref")
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import mlx.core as mx

MODEL_DIR = "/Users/sfk-studio/.lmstudio/models/LoJexLLM/Unlimited-OCR-MLX"

# ============ Part 1: bicubic antialias ============
from ocr_port.model import _resize2d_bicubic_aa

rng = np.random.default_rng(42)
print("=== Part 1: bicubic+antialias vs torch ===")
for (s, t) in [(16, 10), (64, 40), (256, 160), (16, 20)]:
    x = rng.standard_normal((s, s, 8)).astype(np.float32)
    ref = F.interpolate(
        torch.from_numpy(x).permute(2, 0, 1)[None],
        size=(t, t), mode="bicubic", antialias=True, align_corners=False,
    )[0].permute(1, 2, 0).numpy()
    out = np.array(_resize2d_bicubic_aa(mx.array(x), t, t))
    print(f"  {s}->{t}: max|diff| = {np.abs(ref - out).max():.3e}")

# ============ Part 2: 视觉塔全链路 ============
print("\n=== Part 2: vision tower vs official PyTorch (fp32) ===")
import safetensors.torch
import deepencoder

W = safetensors.torch.load_file(os.path.join(MODEL_DIR, "model.safetensors"))

# 官方参考模型
sam = deepencoder.build_sam_vit_b().float().eval()
clip = deepencoder.build_clip_l().float().eval()
from easydict import EasyDict as adict
proj = deepencoder.MlpProjector(adict(projector_type="linear", input_dim=2048, n_embed=1280)).float().eval()

sam_sd = {k[len("sam_model."):]: v.float() for k, v in W.items() if k.startswith("sam_model.")}
missing, unexpected = sam.load_state_dict(sam_sd, strict=True), None
clip_sd = {k[len("vision_model."):]: v.float() for k, v in W.items() if k.startswith("vision_model.")}
clip.load_state_dict(clip_sd, strict=False)  # position_ids 是 buffer 不在权重中
proj_sd = {k[len("projector."):]: v.float() for k, v in W.items() if k.startswith("projector.")}
proj.load_state_dict(proj_sd, strict=True)

image_newline = W["image_newline"].float()
view_seperator = W["view_seperator"].float()

# MLX 模型（fp32，走生产 loader 路径）
from ocr_port.inference import load_model as mlx_load
mlx_model = mlx_load(MODEL_DIR, dtype="float32")

# 构造输入：用真实样张（一张扫描古籍页）
import pymupdf
PDF = "/Volumes/macstudio-work/synology_drive/zotero_attanger/音色研究/古琴音色/张斌_2014_宋代古琴文化考论.pdf"
d = pymupdf.open(PDF)
pix = d[5].get_pixmap(matrix=pymupdf.Matrix(300 / 72, 300 / 72))
img_path = "/tmp/litflow_p0_ocr/cn_scan.png"
pix.save(img_path)
d.close()

from ocr_port.image_processing import load_image, dynamic_preprocess
from PIL import Image, ImageOps

image = load_image(img_path)
# 只取 1 行 2 列 crop（加速 CPU 参考），crop_shape = (w=1, h=2)?? 官方 (i,j)= (cols? rows?)
# dynamic_preprocess 返回 (patches, target_aspect_ratio=(i,j))，i=宽方向块数
patches, crop_shape = dynamic_preprocess(image, min_num=2, max_num=2, image_size=640)
wc, hc = crop_shape
print(f"crop_shape (w={wc}, h={hc}), patches={len(patches)}")

mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

def to_tensor(img):
    a = np.array(img, dtype=np.float32) / 255.0
    return ((a - mean) / std).transpose(2, 0, 1)

global_view = ImageOps.pad(image, (1024, 1024), color=(128, 128, 128))
patches_np = np.stack([to_tensor(p) for p in patches])  # [P,3,640,640]
ori_np = to_tensor(global_view)[None]  # [1,3,1024,1024]

# ---------- 官方 PyTorch 前向 ----------
patches_t = torch.from_numpy(patches_np)
ori_t = torch.from_numpy(ori_np)

def report(name, ref_np, out_mx, ref_layout="nhwc_out"):
    out = np.array(out_mx)
    if out.shape != ref_np.shape:
        # SAM 输出布局差异：torch [B,C,H,W] vs mlx [B,H,W,C]
        if ref_np.ndim == 4 and out.ndim == 4 and ref_np.shape[0] == out.shape[0]:
            ref_np = ref_np.transpose(0, 2, 3, 1)
    d = np.abs(ref_np - out)
    denom = max(np.abs(ref_np).max(), 1e-9)
    print(f"  {name}: shape={out.shape} max|diff|={d.max():.3e} rel={d.max()/denom:.3e}")

with torch.no_grad():
    lf1 = sam(patches_t)           # [P,1024,10,10]
    gf1 = sam(ori_t)               # [1,1024,16,16]
    report("SAM local", lf1.numpy(), mlx_model.sam_model(mx.array(patches_np.transpose(0, 2, 3, 1))))
    report("SAM global", gf1.numpy(), mlx_model.sam_model(mx.array(ori_np.transpose(0, 2, 3, 1))))

    lf2 = clip(patches_t, lf1)     # [P,101,1024]
    gf2 = clip(ori_t, gf1)         # [1,257,1024]
    report("CLIP local", lf2.numpy(), mlx_model.vision_model(mx.array(patches_np.transpose(0, 2, 3, 1)), mlx_model.sam_model(mx.array(patches_np.transpose(0, 2, 3, 1)))))
    report("CLIP global", gf2.numpy(), mlx_model.vision_model(mx.array(ori_np.transpose(0, 2, 3, 1)), mlx_model.sam_model(mx.array(ori_np.transpose(0, 2, 3, 1)))))

    # 最终特征组装（官方 modeling_deepseekocr.forward）
    local_features = torch.cat((lf2[:, 1:], lf1.flatten(2).permute(0, 2, 1)), dim=-1)
    local_features = proj(local_features)
    global_features = torch.cat((gf2[:, 1:], gf1.flatten(2).permute(0, 2, 1)), dim=-1)
    global_features = proj(global_features)

    _, hw, nd = global_features.shape
    h = w = int(hw ** 0.5)
    _2, hw2, nd2 = local_features.shape
    h2 = w2 = int(hw2 ** 0.5)
    width_crop_num, height_crop_num = wc, hc

    gf = global_features.view(h, w, nd)
    gf = torch.cat([gf, image_newline[None, None, :].expand(h, 1, nd)], dim=1)
    gf = gf.view(-1, nd)
    lf = local_features.view(height_crop_num, width_crop_num, h2, w2, nd2).permute(0, 2, 1, 3, 4).reshape(height_crop_num * h2, width_crop_num * w2, nd2)
    lf = torch.cat([lf, image_newline[None, None, :].expand(height_crop_num * h2, 1, nd2)], dim=1)
    lf = lf.view(-1, nd2)
    ref_full = torch.cat([lf, gf, view_seperator[None, :]], dim=0).numpy()

# ---------- MLX 前向（生产路径 encode_images，输入为 NCHW） ----------
images = [
    (
        mx.array(patches_np),  # [P,3,640,640] NCHW
        mx.array(ori_np),      # [1,3,1024,1024] NCHW
    )
]
mlx_full = mlx_model.encode_images(images, images_spatial_crop=[(wc, hc)])[0]
report("final full_feats", ref_full, mlx_full)

print("\n完成。")

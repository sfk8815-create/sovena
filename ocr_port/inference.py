"""Unlimited-OCR MLX Inference Pipeline.

Complete inference pipeline for document OCR using MLX acceleration on Apple Silicon.

Usage:
    python inference.py --model_dir ./unlimited-ocr-mlx-weights --image document.jpg --output ./output
"""

import os
import sys
import json
import time
import argparse
from typing import Optional, List

import numpy as np
import mlx.core as mx

from .config import UnlimitedOCRConfig
from .model import UnlimitedOCRModel
from .image_processing import load_image, preprocess_image, build_input


def load_tokenizer(model_dir: str):
    """Load tokenizer files from the model directory.

    绕开 AutoTokenizer 的 trust_remote_code 路径（transformers 5.x 兼容问题），
    该模型实际使用 LlamaTokenizerFast（DeepSeek 词表风格）。
    """
    from transformers import LlamaTokenizerFast

    tokenizer = LlamaTokenizerFast(
        tokenizer_file=os.path.join(model_dir, "tokenizer.json"),
        bos_token="<｜begin▁of▁sentence｜>",
        eos_token="<｜end▁of▁sentence｜>",
    )
    return tokenizer


# 需 NCHW→NHWC 转置的卷积权重（原权重文件为 PyTorch 布局；neck 键按重命名后判断）
_CONV_KEYS = {
    "sam_model.neck.layers.0.weight", "sam_model.neck.layers.2.weight",
    "sam_model.net_2.weight", "sam_model.net_3.weight",
    "sam_model.patch_embed.proj.weight",
    "vision_model.embeddings.patch_embedding.weight",
}


def load_model(model_dir: str, dtype: str = "float16") -> UnlimitedOCRModel:
    """Load the MLX model with converted weights.

    相对原始移植版的修补：
    1. neck 权重补 `layers.` 中缀（nn.Sequential 命名差异）；
    2. 卷积权重 NCHW→NHWC 转置。
    （窗块 rel_pos 保持 27 长度，与模型槽位及官方实现一致，无需填充。）
    """
    import numpy as np
    import safetensors.torch

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    config = UnlimitedOCRConfig.from_original_config(config_dict)

    weights_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"MLX weights not found at {weights_path}. "
            "Run convert.py first to convert from PyTorch."
        )

    st_weights = safetensors.torch.load_file(weights_path, device="cpu")
    np_dtype = np.float16 if dtype == "float16" else np.float32
    items = []
    for k, v in st_weights.items():
        if k.startswith("sam_model.neck."):
            k = k.replace("sam_model.neck.", "sam_model.neck.layers.")
        a = v.float().numpy()
        if k in _CONV_KEYS:
            a = np.transpose(a, (0, 2, 3, 1))
        if a.dtype == np.float32:
            a = a.astype(np_dtype)
        items.append((k, mx.array(a)))

    model = UnlimitedOCRModel(config)
    # position_ids 是缓冲而非训练权重，从模型树中取出补齐 strict 校验
    tree = model.parameters()
    pid = tree["vision_model"]["embeddings"]["position_ids"]
    items.append(("vision_model.embeddings.position_ids", pid))
    model.load_weights(items)
    # image_newline / view_seperator 为随机初始化缓冲，需与权重同精度以支持 concatenate
    if np_dtype == np.float16:
        import mlx.core as _mx
        model.image_newline = model.image_newline.astype(_mx.float16)
        model.view_seperator = model.view_seperator.astype(_mx.float16)
    mx.eval(model.parameters())

    print(f"Model loaded with {sum(1 for _ in items):,} tensors")
    return model


def create_attention_mask(seq_len: int) -> mx.array:
    """Create causal attention mask."""
    mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
    mask = mx.where(mask, 0.0, float('-inf'))
    return mask[None, None, :, :]


def format_conversation(prompt: str, image_path: str) -> List[dict]:
    """Format conversation for the model."""
    return [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{prompt}",
            "images": [image_path],
        },
        {"role": "Assistant", "content": ""},
    ]


class UnlimitedOCRInference:
    """High-level inference interface for Unlimited-OCR MLX."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load model and tokenizer."""
        print("Loading model...")
        self.model = load_model(self.model_dir)

        print("Loading tokenizer...")
        self.tokenizer = load_tokenizer(self.model_dir)

        print("Ready!")
        return self

    def encode_text(self, text: str, bos: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if bos:
            tokens = [self.tokenizer.bos_token_id] + tokens
        return tokens

    def decode_text(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def process_image(self, image_path: str):
        """Load and preprocess an image."""
        image = load_image(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        return preprocess_image(
            image,
            base_size=1024,
            image_size=640,
            crop_mode=True,
        )

    def infer_single(
        self,
        image_path: str,
        prompt: str = "document parsing.",
        output_dir: Optional[str] = None,
        max_length: int = 32768,
        temperature: float = 0.0,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
    ) -> str:
        """Run OCR inference on a single image.

        Args:
            image_path: Path to the input image
            prompt: OCR prompt
            output_dir: Output directory for results
            max_length: Maximum generation length
            temperature: Sampling temperature (0 = greedy)
            base_size: Base image size for global view
            image_size: Tile size for patches
            crop_mode: Whether to use dynamic tiling

        Returns:
            Generated OCR text
        """
        if self.model is None:
            self.load()

        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

        # 官方 DeepSeek-OCR 'plain' 模板：prompt 即 "<image>document parsing."，
        # 无 User/Assistant 包装。序列布局：
        #   [BOS] + <图像token(128815)×N> + tokenize("document parsing.")
        # 图像特征按 mask 注入，token id 本身不影响被 mask 的位置。
        from .image_processing import load_image as _load
        IMAGE_TOKEN_ID = 128815
        prompt_ids = self.encode_text(prompt, bos=False)

        # Process image
        image = _load(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        patches_arr, orig_arr, crop_shape = preprocess_image(
            image, base_size=base_size, image_size=image_size, crop_mode=crop_mode
        )

        # Convert to MLX arrays
        patches_mx = mx.array(patches_arr) if patches_arr.shape[0] > 0 else None
        orig_mx = mx.array(orig_arr)

        # 图像 token 数（与 encode_images 的特征排布一一对应）：
        #   局部：每行 (10*w_crop + 1 新行符) × (10*h_crop) 行；
        #   全局：16 行 × (16+1) = 272，再加 view_seperator 1
        w_crop, h_crop = crop_shape
        has_local = crop_mode and patches_arr.shape[0] > 0 and (w_crop > 1 or h_crop > 1)
        if has_local:
            n_local = (10 * w_crop + 1) * (10 * h_crop)
            n_image_tokens = n_local + 273
        else:
            n_image_tokens = 273

        extended_ids = (
            [self.tokenizer.bos_token_id]
            + [IMAGE_TOKEN_ID] * n_image_tokens
            + prompt_ids
        )
        image_start = 1
        seq_mask = np.zeros(len(extended_ids), dtype=bool)
        seq_mask[image_start:image_start + n_image_tokens] = True
        total_image_feats = n_image_tokens
        input_ids = extended_ids

        print(f"Input: {len(input_ids)} tokens, {total_image_feats} image tokens")
        print("Running OCR inference...")
        start_time = time.time()

        # Prepare model inputs
        input_ids_mx = mx.array([input_ids], dtype=mx.int32)
        images_seq_mask_mx = mx.array(np.asarray([seq_mask]), dtype=mx.bool_)

        # Prepare image tensor in the format the model expects
        # [patches, original]
        image_tensor = [patches_mx, orig_mx]
        images = [image_tensor]
        images_spatial_crop = [crop_shape] if crop_mode else [(1, 1)]

        # Generate
        output_ids = self.model.generate(
            input_ids=input_ids_mx,
            images=images,
            images_seq_mask=images_seq_mask_mx,
            images_spatial_crop=images_spatial_crop,
            max_length=max_length,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        elapsed = time.time() - start_time
        tokens_generated = output_ids.shape[1] - len(input_ids)
        tps = tokens_generated / elapsed if elapsed > 0 else 0

        # Decode：只解码 prompt 之后新生成的 token，去掉 EOS
        output_tokens = output_ids[0].tolist()[len(input_ids):]
        if output_tokens and output_tokens[-1] == self.tokenizer.eos_token_id:
            output_tokens = output_tokens[:-1]
        result = self.decode_text(output_tokens).strip()

        print(f"\n=== OCR Result ({tokens_generated} tokens, {elapsed:.1f}s, {tps:.1f} t/s) ===")
        print(result)

        if output_dir:
            result_path = os.path.join(output_dir, "result.txt")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"Saved result to {result_path}")

        return result


def main():
    parser = argparse.ArgumentParser(description="Unlimited-OCR MLX Inference")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing MLX weights and tokenizer")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--prompt", type=str, default="document parsing.",
                        help="OCR prompt")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--max_length", type=int, default=32768,
                        help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--base_size", type=int, default=1024,
                        help="Base image size")
    parser.add_argument("--image_size", type=int, default=640,
                        help="Tile image size")
    parser.add_argument("--no_crop", action="store_true",
                        help="Disable dynamic tiling (use base mode)")

    args = parser.parse_args()

    engine = UnlimitedOCRInference(args.model_dir)
    result = engine.infer_single(
        image_path=args.image,
        prompt=args.prompt,
        output_dir=args.output,
        max_length=args.max_length,
        temperature=args.temperature,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=not args.no_crop,
    )


if __name__ == "__main__":
    main()

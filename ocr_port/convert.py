"""Convert PaddlePaddle/Unlimited-OCR PyTorch weights to MLX format.

Usage:
    python convert.py --input_dir ./Unlimited-OCR-original --output_dir ./unlimited-ocr-mlx-weights
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict

import safetensors.torch
import torch

# Add current dir to path for importing the config
sys.path.insert(0, os.path.dirname(__file__))


def load_pytorch_weights(model_dir: str) -> Dict[str, np.ndarray]:
    """Load PyTorch safetensors weights."""
    # Find the model directory containing safetensors
    if os.path.isdir(os.path.join(model_dir, "PaddlePaddle", "Unlimited-OCR")):
        model_dir = os.path.join(model_dir, "PaddlePaddle", "Unlimited-OCR")

    st_path = os.path.join(model_dir, "model-00001-of-000001.safetensors")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"Model file not found: {st_path}")

    print(f"Loading weights from {st_path}...")
    weights = safetensors.torch.load_file(st_path, device="cpu")

    print(f"Loaded {len(weights)} weight tensors")
    return {k: v.float().numpy() for k, v in weights.items()}


def convert_weight_name(pt_name: str) -> str:
    """Convert PyTorch weight name to MLX weight name."""
    # Remove model. prefix
    if pt_name.startswith("model."):
        name = pt_name[6:]  # Remove "model."
    elif pt_name.startswith("lm_head."):
        name = pt_name
    else:
        name = pt_name

    # Map embed_tokens, norm
    if name == "embed_tokens.weight":
        return "language_model.embed_tokens.weight"
    if name == "norm.weight":
        return "language_model.norm.weight"

    # Map lm_head
    if pt_name.startswith("lm_head."):
        return pt_name

    # Map layers
    if name.startswith("layers."):
        parts = name.split(".")
        layer_idx = parts[1]
        rest = ".".join(parts[2:])

        if rest.startswith("self_attn."):
            prefix = "self_attn."
            return f"language_model.layers.{layer_idx}.self_attn.{rest[len(prefix):]}"
        elif rest.startswith("input_layernorm."):
            prefix = "input_layernorm."
            return f"language_model.layers.{layer_idx}.input_layernorm.{rest[len(prefix):]}"
        elif rest.startswith("post_attention_layernorm."):
            prefix = "post_attention_layernorm."
            return f"language_model.layers.{layer_idx}.post_attention_layernorm.{rest[len(prefix):]}"
        elif rest.startswith("mlp."):
            mlp_rest = rest[4:]  # Remove "mlp."
            if mlp_rest.startswith("gate.weight"):
                return f"language_model.layers.{layer_idx}.mlp.gate.weight"
            elif mlp_rest.startswith("shared_experts."):
                return f"language_model.layers.{layer_idx}.mlp.shared_experts.{mlp_rest[15:]}"
            elif mlp_rest.startswith("experts."):
                return f"language_model.layers.{layer_idx}.mlp.experts.{mlp_rest[8:]}"
            else:
                return f"language_model.layers.{layer_idx}.mlp.{mlp_rest}"

    # Map SAM model
    if name.startswith("sam_model."):
        return name

    # Map vision model (CLIP)
    if name.startswith("vision_model."):
        return name

    # Map projector
    if name.startswith("projector."):
        return name

    # Map image special tokens
    if name in ["image_newline", "view_seperator"]:
        return name

    print(f"WARNING: Unmapped weight: {pt_name}")
    return pt_name


def convert_weights_to_mlx(pt_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert all weights to MLX-compatible format."""
    mlx_weights = {}

    for pt_name, value in pt_weights.items():
        mlx_name = convert_weight_name(pt_name)
        mlx_weights[mlx_name] = value

    print(f"Converted {len(mlx_weights)} weights to MLX format")
    return mlx_weights


def save_mlx_weights(weights: Dict[str, np.ndarray], output_dir: str):
    """Save MLX weights in safetensors format."""
    os.makedirs(output_dir, exist_ok=True)

    # Save as safetensors (MLX compatible)
    output_path = os.path.join(output_dir, "model.safetensors")
    torch_weights = {k: torch.from_numpy(v.copy()) for k, v in weights.items()}
    safetensors.torch.save_file(torch_weights, output_path)
    print(f"Saved weights to {output_path}")

    # Also save config
    import json as j
    config = {
        "model_type": "unlimited-ocr-mlx",
        "architectures": ["UnlimitedOCRModel"],
        "vocab_size": 129280,
        "hidden_size": 1280,
        "num_hidden_layers": 12,
        "num_attention_heads": 10,
        "num_key_value_heads": 10,
        "head_dim": 128,
        "intermediate_size": 6848,
        "moe_intermediate_size": 896,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "num_experts_per_tok": 6,
        "first_k_dense_replace": 1,
        "max_position_embeddings": 32768,
        "vision_output_dim": 2048,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        j.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Unlimited-OCR to MLX format")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing original PyTorch model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for MLX weights")
    args = parser.parse_args()

    print("=== Unlimited-OCR: PyTorch → MLX Weight Converter ===\n")

    # Load original weights
    pt_weights = load_pytorch_weights(args.input_dir)

    # Convert
    mlx_weights = convert_weights_to_mlx(pt_weights)

    # Save
    save_mlx_weights(mlx_weights, args.output_dir)

    # Print summary
    total_params = sum(v.size for v in mlx_weights.values())
    print(f"\nDone! Total parameters: {total_params:,} ({total_params * 2 / 1e9:.2f}B BF16)")

    # Copy tokenizer files
    input_model_dir = args.input_dir
    if os.path.isdir(os.path.join(input_model_dir, "PaddlePaddle", "Unlimited-OCR")):
        input_model_dir = os.path.join(input_model_dir, "PaddlePaddle", "Unlimited-OCR")

    import shutil
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(input_model_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(args.output_dir, fname)
            shutil.copy2(src, dst)
            print(f"Copied {fname}")


if __name__ == "__main__":
    main()

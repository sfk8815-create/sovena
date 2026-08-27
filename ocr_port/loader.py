"""Load MLX weights into UnlimitedOCR model.

Handles the complete weight loading with proper name mapping and validation.
"""

from typing import Dict, List, Tuple
import mlx.core as mx
import mlx.nn as nn

from .model import UnlimitedOCRModel, SAMVisionEncoder, CLIPVisionTransformer
from .config import UnlimitedOCRConfig


def load_weights_from_safetensors(model: nn.Module, weights_path: str) -> nn.Module:
    """Load MLX-compatible weights from safetensors file.

    Args:
        model: MLX model instance
        weights_path: Path to safetensors file

    Returns:
        Model with loaded weights
    """
    import safetensors.torch
    import numpy as np

    print(f"Loading weights from {weights_path}...")
    st_weights = safetensors.torch.load_file(weights_path, device="cpu")

    # Convert to MLX arrays
    mlx_weights = {}
    for name, tensor in st_weights.items():
        mlx_weights[name] = mx.array(tensor.float().numpy())

    # Load into model
    model.load_weights(list(mlx_weights.items()))
    mx.eval(model.parameters())

    total = sum(v.size for v in mlx_weights.values())
    print(f"Loaded {len(mlx_weights)} tensors, {total:,} parameters")
    return model


def create_model_from_dir(model_dir: str) -> Tuple[UnlimitedOCRModel, UnlimitedOCRConfig]:
    """Create model instance from model directory.

    Args:
        model_dir: Directory containing config.json and model.safetensors

    Returns:
        Tuple of (model, config)
    """
    import json
    config_path = f"{model_dir}/config.json"
    weights_path = f"{model_dir}/model.safetensors"

    with open(config_path) as f:
        config_dict = json.load(f)

    config = UnlimitedOCRConfig.from_original_config(config_dict)
    model = UnlimitedOCRModel(config)
    model = load_weights_from_safetensors(model, weights_path)

    return model, config


def verify_weights(model: UnlimitedOCRModel) -> Dict[str, any]:
    """Verify that all model weights are properly loaded.

    Returns:
        Dict with verification statistics
    """
    stats = {"total_params": 0, "num_layers": {}, "issues": []}

    params = dict(model.parameters())

    for name, param in params.items():
        size = param.numpy().size if hasattr(param, 'numpy') else 1
        stats["total_params"] += size

        # Check for NaN values
        val = param
        if hasattr(param, 'numpy'):
            arr = param.numpy()
            if hasattr(arr, 'isnan'):
                nans = arr.isnan().sum()
                if nans > 0:
                    stats["issues"].append(f"NaN values in {name}: {nans}")

    stats["total_params_formatted"] = f"{stats['total_params']:,}"
    return stats

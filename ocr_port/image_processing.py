"""Image preprocessing for Unlimited-OCR compatible with MLX.

Handles image loading, tiling, normalization, and batch preparation.
"""

import math
from typing import List, Tuple, Optional
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps


def load_image(image_path: str) -> Optional[Image.Image]:
    """Load an image with EXIF orientation correction."""
    try:
        image = Image.open(image_path)
        corrected = ImageOps.exif_transpose(image)
        return corrected.convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """Find the closest allowed aspect ratio for tiling."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 2,
    max_num: int = 32,
    image_size: int = 640,
    use_thumbnail: bool = False,
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """Dynamically tile an image into patches.

    Args:
        image: Input PIL image
        min_num: Minimum number of patches
        max_num: Maximum number of patches
        image_size: Size of each patch
        use_thumbnail: Whether to include a thumbnail

    Returns:
        Tuple of (list of patch images, aspect ratio)
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate valid aspect ratios
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if min_num <= i * j <= max_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find best ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and crop patches
    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        col = i % target_aspect_ratio[0]
        row = i // target_aspect_ratio[0]
        box = (
            col * image_size,
            row * image_size,
            (col + 1) * image_size,
            (row + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images, target_aspect_ratio


def preprocess_image(
    image: Image.Image,
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple], np.ndarray]:
    """Preprocess an image for the model.

    Args:
        image: Input PIL image
        base_size: Base image size for the global view (1024)
        image_size: Tile size for patches (640)
        crop_mode: Whether to use dynamic tiling

    Returns:
        Tuple of (patches_array, original_array, crop_shape, num_image_tokens)
    """
    # Normalize transform
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def to_tensor(img: Image.Image) -> np.ndarray:
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        return arr.transpose(2, 0, 1)  # [C, H, W]

    if crop_mode:
        # 官方：小图（≤640×640）不裁切；大图 dynamic_preprocess（默认 max_num=9）
        if image.size[0] <= image_size and image.size[1] <= image_size:
            crop_shape = (1, 1)
            patches_arr = np.zeros((0, 3, image_size, image_size), dtype=np.float32)
        else:
            patches, crop_shape = dynamic_preprocess(
                image, min_num=2, max_num=9, image_size=image_size
            )
            patches_arr = np.stack([to_tensor(p) for p in patches], axis=0)  # [N, 3, 640, 640]

        # Global view：官方用 ImageOps.pad 保持长宽比、灰底(128,128,128)
        orig_img = ImageOps.pad(image, (base_size, base_size), color=(128, 128, 128))
        orig_arr = to_tensor(orig_img)[np.newaxis, ...]  # [1, 3, 1024, 1024]

        # Number of image tokens
        n_patches = len(patches)
        local_tokens = n_patches * (image_size // 16) ** 2  # Each patch → 40*40 area
        # After SAM: 40*40=1600 tokens per patch → CLIP processes them
        # After CLIP: 256 tokens per patch (1024/4=256?)
        # Let's compute from the architecture: image_size=640, patch=16 → 40x40=1600
        # SAM output: 1024-dim, 16x16 spatial (net_3 output)
        # CLIP output: concat [CLIP[:, 1:], SAM flatten] → 2048, 256 spatial

        # For seq_mask: each image patch contributes a region
        # The model handles this internally; we just need to track the crop shape
        return patches_arr, orig_arr, crop_shape

    else:
        # Single image without tiling (base mode)
        orig_img = ImageOps.pad(image, (base_size, base_size), color=(128, 128, 128))
        orig_arr = to_tensor(orig_img)[np.newaxis, ...]  # [1, 3, 1024, 1024]

        # No patches
        patches_arr = np.zeros((0, 3, image_size, image_size), dtype=np.float32)
        return patches_arr, orig_arr, (1, 1)


def build_input(
    input_ids: List[int],
    image_features_count: int,
) -> Tuple[List[int], np.ndarray]:
    """Build input with image placeholder tokens.

    Args:
        input_ids: Text token ids
        image_features_count: Number of image feature vectors to insert

    Returns:
        Tuple of (extended_input_ids, seq_mask)
    """
    # Image placeholder tokens use token ID 0 (or special image token)
    IMAGE_TOKEN_ID = 0  # This model uses BOS token as placeholder

    # Insert image tokens after the image placeholder in the prompt
    # The model's conversation format uses <image> tag
    extended_ids = []
    seq_mask = []  # True where image features go

    i = 0
    while i < len(input_ids):
        extended_ids.append(input_ids[i])
        seq_mask.append(False)

        # After BOS (token 0), insert image placeholder positions
        if input_ids[i] == 0 and image_features_count > 0:
            # Extend with image placeholder positions
            for _ in range(image_features_count):
                extended_ids.append(0)
                seq_mask.append(True)
            image_features_count = 0  # Only insert once

        i += 1

    return extended_ids, np.array(seq_mask, dtype=bool)

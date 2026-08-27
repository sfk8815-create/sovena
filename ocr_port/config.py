"""Unlimited-OCR Model Configuration for MLX."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class VisionConfig:
    """Vision encoder configuration."""
    # SAM-ViT-B
    sam_img_size: int = 1024
    sam_patch_size: int = 16
    sam_embed_dim: int = 768
    sam_depth: int = 12
    sam_num_heads: int = 12
    sam_mlp_ratio: float = 4.0
    sam_out_chans: int = 256
    sam_window_size: int = 14
    sam_global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11)

    # CLIP-L ViT
    clip_hidden_size: int = 1024
    clip_num_layers: int = 24
    clip_num_heads: int = 16
    clip_ffn_hidden_size: int = 4096
    clip_image_size: int = 224
    clip_patch_size: int = 14
    clip_seq_length: int = 256
    clip_layernorm_epsilon: float = 1e-5

    # Combined output
    vision_output_dim: int = 2048  # SAM(1024) + CLIP(1024)


@dataclass
class LanguageConfig:
    """DeepSeek-V2 language model configuration."""
    vocab_size: int = 129280
    hidden_size: int = 1280
    intermediate_size: int = 6848
    moe_intermediate_size: int = 896
    num_hidden_layers: int = 12
    num_attention_heads: int = 10
    num_key_value_heads: int = 10
    head_dim: int = 128

    # MoE
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    num_experts_per_tok: int = 6
    first_k_dense_replace: int = 1  # Layer 0 is dense
    topk_method: str = "greedy"
    scoring_func: str = "softmax"
    norm_topk_prob: bool = False

    # RoPE
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0

    # Norm
    rms_norm_eps: float = 1e-6

    # Special tokens
    bos_token_id: int = 0
    eos_token_id: int = 1

    # Other
    hidden_act: str = "silu"
    sliding_window_size: int = 128


@dataclass
class ProjectorConfig:
    """Vision-to-language projector configuration."""
    input_dim: int = 2048
    n_embed: int = 1280
    projector_type: str = "linear"


@dataclass
class UnlimitedOCRConfig:
    """Complete Unlimited-OCR configuration."""
    vision: VisionConfig = field(default_factory=VisionConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)

    # Image processing
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True
    min_crops: int = 2
    max_crops: int = 32
    candidate_resolutions: List[List[int]] = field(default_factory=lambda: [[1024, 1024]])

    # Generation
    max_length: int = 32768
    temperature: float = 0.0
    no_repeat_ngram_size: int = 35
    ngram_window: int = 128

    model_type: str = "unlimited-ocr-mlx"

    @classmethod
    def from_original_config(cls, config_dict: dict) -> "UnlimitedOCRConfig":
        """Create config from original PyTorch config.json."""
        lang_cfg = config_dict.get("language_config", config_dict)

        vision = VisionConfig(
            sam_img_size=config_dict.get("vision_config", {}).get("image_size", 1024),
        )

        language = LanguageConfig(
            vocab_size=lang_cfg.get("vocab_size", 129280),
            hidden_size=lang_cfg.get("hidden_size", 1280),
            intermediate_size=lang_cfg.get("intermediate_size", 6848),
            moe_intermediate_size=lang_cfg.get("moe_intermediate_size", 896),
            num_hidden_layers=lang_cfg.get("num_hidden_layers", 12),
            num_attention_heads=lang_cfg.get("num_attention_heads", 10),
            num_key_value_heads=lang_cfg.get("num_key_value_heads", 10),
            n_routed_experts=lang_cfg.get("n_routed_experts", 64),
            n_shared_experts=lang_cfg.get("n_shared_experts", 2),
            num_experts_per_tok=lang_cfg.get("num_experts_per_tok", 6),
            first_k_dense_replace=lang_cfg.get("first_k_dense_replace", 1),
            max_position_embeddings=lang_cfg.get("max_position_embeddings", 32768),
            rope_theta=lang_cfg.get("rope_theta", 10000.0),
            rms_norm_eps=lang_cfg.get("rms_norm_eps", 1e-6),
            bos_token_id=lang_cfg.get("bos_token_id", 0),
            eos_token_id=lang_cfg.get("eos_token_id", 1),
            sliding_window_size=lang_cfg.get("sliding_window_size", 128),
        )

        proj_cfg = config_dict.get("projector_config", {})
        projector = ProjectorConfig(
            input_dim=proj_cfg.get("input_dim", 2048),
            n_embed=proj_cfg.get("n_embed", 1280),
            projector_type=proj_cfg.get("projector_type", "linear"),
        )

        return cls(vision=vision, language=language, projector=projector)

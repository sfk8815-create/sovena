# Unlimited-OCR MLX Implementation
# High-precision OCR model optimized for Apple Silicon via MLX framework
#
# 来源与许可：模型本体为百度开源的 Unlimited-OCR（MIT，
# https://github.com/baidu/Unlimited-OCR ，权重 https://huggingface.co/baidu/Unlimited-OCR ）；
# 本目录代码移植自 mlx-vlm 社区的 MLX 实现（MIT，https://github.com/Blaizzy/mlx-vlm ）。
from .config import UnlimitedOCRConfig
from .model import UnlimitedOCRModel
from .inference import UnlimitedOCRInference

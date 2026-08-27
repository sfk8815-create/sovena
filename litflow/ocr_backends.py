"""litflow OCR 后端抽象。

OCR 通道支持两种后端，输出统一为 Unlimited-OCR 的结构化行格式
（`kind [x1,y1,x2,y2]text`），下游 converter 无感知：

  1. mlx  —— 本仓库 ocr_port/（Apple Silicon 专用，权重经 LM Studio 下载）
  2. http —— OpenAI 兼容接口（llama-server / LM Studio / vLLM 等
             服务 Unlimited-OCR 的 GGUF 量化版），任意平台可用（含
             Windows / Linux / 纯 CPU）

后端选择（环境变量）：
  LITFLOW_OCR_BACKEND = mlx | http | auto（默认 auto：设置了
  LITFLOW_OCR_API 就用 http，否则 mlx）

http 后端相关变量：
  LITFLOW_OCR_API        服务地址（如 http://localhost:8080/v1）
  LITFLOW_OCR_MODEL_NAME 模型名（默认 Unlimited-OCR）
  LITFLOW_OCR_API_KEY    可选 bearer key
"""
from __future__ import annotations

import base64
import os
from typing import Optional

import httpx

DEFAULT_OCR_API = os.environ.get("LITFLOW_OCR_API", "")
DEFAULT_OCR_MODEL_NAME = os.environ.get("LITFLOW_OCR_MODEL_NAME", "Unlimited-OCR")
DEFAULT_OCR_API_KEY = os.environ.get("LITFLOW_OCR_API_KEY", "")


class OpenAICompatOCREngine:
    """经 OpenAI 兼容 chat/completions 接口做 OCR（服务端跑 GGUF 等）。

    接口与 ocr_port.UnlimitedOCRInference 对齐（infer_single），
    converter 无需改动。
    """

    def __init__(
        self,
        api_base: str = DEFAULT_OCR_API,
        model: str = DEFAULT_OCR_MODEL_NAME,
        api_key: str = DEFAULT_OCR_API_KEY,
        timeout: float = 600.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def infer_single(
        self,
        image_path: str,
        prompt: str = "document parsing.",
        max_length: int = 4096,
    ) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        # 常见位图格式均可
        mime = "image/png" if image_path.lower().endswith((".png",)) else "image/jpeg"
        payload = {
            "model": self.model,
            "max_tokens": max_length,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = httpx.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"OCR 服务返回异常: {data}") from e


def resolve_backend() -> str:
    """决定 OCR 后端：显式 LITFLOW_OCR_BACKEND 优先，否则按是否配置 API 自动。"""
    choice = (os.environ.get("LITFLOW_OCR_BACKEND") or "auto").lower()
    if choice in ("mlx", "http"):
        return choice
    return "http" if DEFAULT_OCR_API else "mlx"


def backend_info() -> dict:
    """当前 OCR 后端配置（供 Web 设置页 / doctor 展示）。"""
    backend = resolve_backend()
    return {
        "backend": backend,
        "api": DEFAULT_OCR_API or None,
        "model_name": DEFAULT_OCR_MODEL_NAME if backend == "http" else None,
        "mlx_model_dir": os.path.expanduser(
            os.environ.get(
                "LITFLOW_OCR_MODEL",
                "~/.lmstudio/models/LoJexLLM/Unlimited-OCR-MLX",
            )
        )
        if backend == "mlx"
        else None,
    }

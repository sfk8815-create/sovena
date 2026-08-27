"""P0-4b 诊断：纯文本补全，隔离语言模型（不喂图）。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ocr_port import UnlimitedOCRInference

MODEL_DIR = "/Users/sfk-studio/.lmstudio/models/LoJexLLM/Unlimited-OCR-MLX"

inf = UnlimitedOCRInference(MODEL_DIR)
inf.model = None
print("Loading model...")
from ocr_port.inference import load_model, load_tokenizer
import mlx.core as mx
inf.model = load_model(MODEL_DIR)
inf.tokenizer = load_tokenizer(MODEL_DIR)

prompts = [
    "The capital of France is",
    "古琴是中国传统的弹拨乐器，",
]

for p in prompts:
    ids = inf.encode_text(p, bos=True)
    input_ids = mx.array([ids], dtype=mx.int32)
    out = inf.model.generate(input_ids=input_ids, max_length=60, temperature=0.0, eos_token_id=1)
    toks = out[0].tolist()[len(ids):]
    if toks and toks[-1] == 1:
        toks = toks[:-1]
    text = inf.tokenizer.decode(toks, skip_special_tokens=True)
    print(f"\n>>> {p}\n    {text!r}")

"""P0-4 终版：使用 vendored ocr_port 跑 OCR 样张验证。"""
import sys, os, time
import pymupdf

MODEL_DIR = "/Users/sfk-studio/.lmstudio/models/LoJexLLM/Unlimited-OCR-MLX"
OUT = "/tmp/litflow_p0_ocr"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(__file__))

from ocr_port import UnlimitedOCRInference

samples = {
    "cn_scan": ("/Volumes/macstudio-work/synology_drive/zotero_attanger/音色研究/古琴音色/张斌_2014_宋代古琴文化考论.pdf", 5),
    "en_text": ("/Volumes/macstudio-work/synology_drive/zotero_attanger/音色研究/古琴音色/Waltham 等_2017_Acoustics of the qin.pdf", 2),
}
img_paths = {}
for name, (pdf, pg) in samples.items():
    d = pymupdf.open(pdf)
    pix = d[pg].get_pixmap(matrix=pymupdf.Matrix(300/72, 300/72))
    p = f"{OUT}/{name}.png"
    pix.save(p)
    img_paths[name] = p
    d.close()
    print(f"样张 {name}: {pix.width}x{pix.height}")

t0 = time.time()
inf = UnlimitedOCRInference(MODEL_DIR).load()
print(f"模型加载: {time.time()-t0:.1f}s")

for name, p in img_paths.items():
    t0 = time.time()
    text = inf.infer_single(p, prompt="document parsing.", max_length=4096)
    dt = time.time() - t0
    print(f"\n===== {name}（{dt:.1f}s, {len(text)} 字符）=====")
    print(text[:600])
    with open(f"{OUT}/{name}.md", "w") as f:
        f.write(text)
print("\n完成，输出在", OUT)

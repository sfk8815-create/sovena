"""P0 验证：anydoc 转换 + LM Studio embedding + 文本层检测"""
import sys, time, json, urllib.request

PDF = "/Volumes/macstudio-work/synology_drive/zotero_attanger/音色研究/古琴音色/Waltham 等_2017_Acoustics of the qin.pdf"

# ── 1. anydoc ──
print("== anydoc ==")
import anydoc
t0 = time.time()
md = anydoc.to_markdown(PDF)
print(f"markdown 长度: {len(md)} 字符, 耗时 {time.time()-t0:.2f}s")
print("前 500 字符:")
print(md[:500])
print("...")

# ── 2. pymupdf 文本层检测（通道分流的判定基础）──
print("\n== pymupdf 文本层 ==")
import pymupdf
doc = pymupdf.open(PDF)
print(f"页数: {len(doc)}, page labels: {doc[0].get_label()} .. {doc[-1].get_label()}")
t1 = doc[0].get_text()
print(f"第1页文本层字符数: {len(t1)}")

# ── 3. LM Studio embedding ──
print("\n== embedding ==")
def embed(texts):
    body = json.dumps({"model": "text-embedding-qwen3-embedding-4b", "input": texts}).encode()
    req = urllib.request.Request("http://localhost:1234/v1/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return d["data"]

t0 = time.time()
texts = ["古琴的音色具有独特的衰减特性", "The qin exhibits distinctive timbral decay characteristics", "和声学教程第三章"]
out = embed(texts)
dims = [len(e["embedding"]) for e in out]
print(f"批数: {len(out)}, 维度: {dims}, 耗时 {time.time()-t0:.2f}s")
# 语义相似度抽检（中英互译应相近）
import math
v1, v2, v3 = [e["embedding"] for e in out]
def cos(a, b):
    s = sum(x*y for x, y in zip(a, b))
    return s / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))
print(f"cos(中文古琴句, 英文qin句) = {cos(v1, v2):.4f}  (应较高)")
print(f"cos(中文古琴句, 和声学)   = {cos(v1, v3):.4f}  (应较低)")
print("\nP0 验证全部完成")

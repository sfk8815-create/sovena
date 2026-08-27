"""litflow 文献转换器：L1 文本路 / L2 OCR 路 / anydoc 非PDF路。

所有输出为对 AI 友好的 markdown，并尽可能标注【书页页码】（PDF Page Label，
非物理页序）：
  - L1：pymupdf 逐页文本块提取，页标签来自 PDF Page Labels
  - L2：pymupdf 渲染位图 → Unlimited-OCR-MLX 结构化识别，
        页标签优先级：OCR 识别的 page_number > PDF Page Label > 物理页序
  - 非PDF（docx/epub/html/快照等）：anydoc / trafilatura
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import pymupdf

try:
    from anydoc import to_markdown as _anydoc_to_markdown
except ImportError:  # pragma: no cover
    _anydoc_to_markdown = None

try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None


OCR_LINE_RE = re.compile(r"^(?P<kind>[a-z_]+)\s*\[(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\](?P<text>.*)$")

# OCR 行类型 → markdown 前缀
OCR_KIND_PREFIX = {
    "title": "## ",
    "header": None,       # 页眉，丢弃
    "footer": None,       # 页脚，丢弃
    "page_number": None,  # 页码，单独处理
    "aside_text": None,   # 侧栏文字（古籍边栏），丢弃
    "text": "",
    "table": "",
    "figure": "",
    "formula": "$$",
}


@dataclass
class ConvertResult:
    markdown: str
    route: str                    # text | ocr | anydoc | web | empty
    pages: int = 0
    page_labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.markdown)


# ---------------------------------------------------------------------------
# 页码标签
# ---------------------------------------------------------------------------

def page_label(doc: pymupdf.Document, i: int) -> str:
    """第 i 页（0 基）的页码标签；无标签时回退物理页序。"""
    try:
        label = doc[i].get_label()
    except Exception:
        label = ""
    return label or str(i + 1)


# ---------------------------------------------------------------------------
# PDF 分级：文本覆盖启发式
# ---------------------------------------------------------------------------

def classify_pdf(path: str, sample_pages: int = 10, min_chars: int = 200) -> str:
    """返回 'text'（走 L1）或 'ocr'（走 L2）。

    抽样前若干页统计可提取文本量；影印/扫描件基本无文本层。
    """
    doc = pymupdf.open(path)
    try:
        n = min(len(doc), sample_pages)
        if n == 0:
            return "text"
        total = 0
        for i in range(n):
            total += len(doc[i].get_text().strip())
        return "text" if total / n >= min_chars else "ocr"
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# L1：文本路
# ---------------------------------------------------------------------------

def pdf_text_to_markdown(path: str) -> ConvertResult:
    doc = pymupdf.open(path)
    labels: list[str] = []
    parts: list[str] = []
    try:
        for i in range(len(doc)):
            label = page_label(doc, i)
            labels.append(label)
            page = doc[i]
            blocks = [
                b for b in page.get_text("blocks")
                if b[6] == 0 and b[4].strip()  # 文本块，非空
            ]
            blocks.sort(key=lambda b: (round(b[1], 1), b[0]))
            texts = [b[4].strip() for b in blocks]
            if texts:
                parts.append(f"\n\n**[p.{label}]**\n\n" + "\n\n".join(texts))
        md = "".join(parts).strip()
        return ConvertResult(
            markdown=md,
            route="text",
            pages=len(doc),
            page_labels=labels,
        )
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# L2：OCR 路
# ---------------------------------------------------------------------------

def _parse_ocr_lines(raw: str) -> list[dict]:
    lines = []
    for ln in raw.splitlines():
        m = OCR_LINE_RE.match(ln.strip())
        if not m:
            if ln.strip() and lines:
                # 续行：拼接到上一条
                lines[-1]["text"] += " " + ln.strip()
            continue
        lines.append(
            {
                "kind": m.group("kind"),
                "box": tuple(int(m.group(k)) for k in (2, 3, 4, 5)),
                "text": m.group("text").strip(),
            }
        )
    return lines


def _ocr_page_markdown(lines: list[dict]) -> tuple[str, Optional[str]]:
    """OCR 行 → markdown 正文；返回 (markdown, 检测到的印刷页码)。"""
    page_number = None
    body: list[str] = []
    for ln in lines:
        kind, text = ln["kind"], ln["text"]
        if not text:
            continue
        if kind == "page_number":
            page_number = text
            continue
        prefix = OCR_KIND_PREFIX.get(kind, "")
        if prefix is None:
            continue
        if kind == "title":
            body.append(f"## {text}")
        else:
            body.append(text)
    return "\n\n".join(body), page_number


def pdf_ocr_to_markdown(
    path: str,
    ocr_engine,
    dpi: int = 300,
    max_tokens: int = 4096,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ConvertResult:
    """ocr_engine：已加载的 UnlimitedOCRInference 实例。"""
    doc = pymupdf.open(path)
    labels: list[str] = []
    parts: list[str] = []
    notes: list[str] = []
    zoom = dpi / 72
    try:
        n = len(doc)
        for i in range(n):
            pix = doc[i].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
            img_path = f"/tmp/litflow_ocr_{i}.png"
            pix.save(img_path)
            try:
                raw = ocr_engine.infer_single(
                    img_path, prompt="document parsing.", max_length=max_tokens
                )
            finally:
                if os.path.exists(img_path):
                    os.remove(img_path)
            lines = _parse_ocr_lines(raw)
            body, detected = _ocr_page_markdown(lines)
            label = detected or page_label(doc, i)
            labels.append(label)
            if detected and detected != page_label(doc, i):
                notes.append(f"p{i + 1}: OCR 页码 {detected}（PDF 标签 {page_label(doc, i) or '无'}）")
            if body:
                parts.append(f"\n\n**[p.{label}]**\n\n{body}")
            if progress:
                progress(i + 1, n)
        md = "".join(parts).strip()
        return ConvertResult(markdown=md, route="ocr", pages=n, page_labels=labels, notes=notes)
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# 非 PDF：anydoc / 网页
# ---------------------------------------------------------------------------

def anydoc_to_markdown(path: str) -> ConvertResult:
    if _anydoc_to_markdown is None:
        raise ImportError("firecrawl-anydoc 未安装")
    md = _anydoc_to_markdown(path)
    if isinstance(md, bytes):
        md = md.decode("utf-8", errors="ignore")
    return ConvertResult(markdown=md.strip(), route="anydoc")


def html_to_markdown(html_path: str) -> ConvertResult:
    """本地网页快照 → 正文 markdown（trafilatura）。"""
    if trafilatura is None:
        raise ImportError("trafilatura 未安装")
    with open(html_path, encoding="utf-8", errors="ignore") as f:
        html = f.read()
    extracted = trafilatura.extract(html, output_format="markdown", include_comments=False)
    md = extracted or html
    return ConvertResult(markdown=md.strip(), route="anydoc")


def url_to_markdown(url: str, fetch: Optional[Callable[[str], str]] = None, timeout: float = 20.0) -> ConvertResult:
    """在线 URL → 正文 markdown。fetch 可注入（默认带超时的抓取）。"""
    if trafilatura is None:
        raise ImportError("trafilatura 未安装")
    if fetch is not None:
        html = fetch(url)
    else:
        import httpx

        try:
            r = httpx.get(
                url,
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            )
            html = r.text if r.status_code == 200 else None
        except Exception:
            html = None
    if not html:
        return ConvertResult(markdown="", route="web", notes=[f"抓取失败: {url}"])
    extracted = trafilatura.extract(html, output_format="markdown", include_comments=False)
    return ConvertResult(markdown=(extracted or "").strip(), route="web")


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def convert_attachment(
    path: Optional[str],
    content_type: Optional[str] = None,
    url: Optional[str] = None,
    ocr_engine=None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ConvertResult:
    """按附件类型自动路由转换。"""
    if path and os.path.isfile(path):
        if content_type == "application/pdf" or path.lower().endswith(".pdf"):
            if classify_pdf(path) == "text":
                return pdf_text_to_markdown(path)
            if ocr_engine is None:
                return ConvertResult(
                    markdown="", route="ocr",
                    notes=["需要 OCR 引擎但未提供（调用方应先加载）"],
                )
            return pdf_ocr_to_markdown(path, ocr_engine, progress=progress)
        if path.lower().endswith((".html", ".htm", ".xhtml")):
            return html_to_markdown(path)
        if path.lower().endswith((".txt", ".md", ".markdown", ".csv")):
            with open(path, encoding="utf-8", errors="ignore") as f:
                return ConvertResult(markdown=f.read().strip(), route="text")
        try:
            return anydoc_to_markdown(path)
        except Exception as e:  # anydoc 不支持的格式
            return ConvertResult(markdown="", route="empty", notes=[f"anydoc 失败: {e}"])
    if url:
        return url_to_markdown(url)
    return ConvertResult(markdown="", route="empty", notes=["无可转换内容"])

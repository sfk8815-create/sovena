"""sovena 临时资料处理（ad-hoc）：把任意本地文件/文件夹做成可检索语义包。

适用场景：非 Zotero 资料（如 E_book 电子书库、散装 PDF、讲义等）
批量转 markdown + 向量索引，之后与 Zotero 分类一样可被自然语言检索。

    python -m sovena.cli adhoc /path/to/dir --name 我的书库
    （或经 Web UI / MCP 提交）

目录结构：
  <SOVENA_ROOT>/adhoc/<名称>/
    _manifest.json
    <文件名slug>/
      meta.json
      content.md
向量集合名：adhoc:<名称>
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

from .converter import classify_pdf, convert_attachment
from .packager import file_fingerprint, md5_of_file

# anydoc / converter 可处理的扩展名（排除图片：走 converter 无图像路）
SUPPORTED_EXTS = {
    ".pdf", ".epub", ".docx", ".doc", ".html", ".htm", ".xhtml",
    ".txt", ".md", ".markdown", ".rtf", ".odt", ".xlsx", ".pptx", ".csv",
}
EXCLUDE_DIRS = {".git", "__pycache__", ".DS_Store", "node_modules", "_lancedb"}
EXCLUDE_FILES = {"_manifest.json", "meta.json", "content.md"}

ProgressFn = Callable[[str, dict], None]


def _slug(text: str, max_len: int = 70) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\\/:*?\"<>|]+", " ", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len].strip("_") or "untitled"


@dataclass
class AdhocFile:
    path: str
    size: int
    item_key: str  # 稳定 ID（源路径 md5 前 12 位），用于向量库增删

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def ext(self) -> str:
        return os.path.splitext(self.path)[1].lower()


def stable_key(path: str) -> str:
    import hashlib

    return "adh" + hashlib.md5(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]


def scan_files(paths: list[str], recursive: bool = True) -> list[AdhocFile]:
    """展开路径（文件/目录）为待处理文件列表。"""
    out: dict[str, AdhocFile] = {}
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isfile(p):
            _add(out, p)
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                if not recursive:
                    dirs[:] = []
                for f in sorted(files):
                    if f.startswith(".") or f in EXCLUDE_FILES:
                        continue
                    _add(out, os.path.join(root, f))
        else:
            raise FileNotFoundError(f"路径不存在: {p}")
    return sorted(out.values(), key=lambda f: f.path)


def _add(out: dict, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        return
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    out[os.path.abspath(path)] = AdhocFile(
        path=os.path.abspath(path), size=size, item_key=stable_key(path)
    )


# ---------------------------------------------------------------------------
# 处理器
# ---------------------------------------------------------------------------

@dataclass
class AdhocSummary:
    collection: str
    files: int = 0
    converted: int = 0
    skipped: int = 0
    empty: int = 0
    ocr_used: bool = False
    chunks_indexed: int = 0
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "collection": self.collection,
            "files": self.files,
            "converted": self.converted,
            "skipped": self.skipped,
            "empty": self.empty,
            "ocr_used": self.ocr_used,
            "chunks_indexed": self.chunks_indexed,
            "elapsed_sec": self.elapsed_sec,
        }


class AdhocProcessor:
    """临时资料处理器：转换 + 落盘 + 向量索引（复用 Pipeline 的组件）。"""

    def __init__(self, packager, index, embedder, ocr_holder):
        self.packager = packager
        self.index = index
        self.embedder = embedder
        self.ocr = ocr_holder

    @property
    def root(self) -> str:
        return os.path.join(self.packager.root, "adhoc")

    def adhoc_dir(self, name: str) -> str:
        return os.path.join(self.root, _slug(name, max_len=80))

    def collection_name(self, name: str) -> str:
        return f"adhoc:{name}"

    def list_packs(self) -> list[dict]:
        """列出已有 adhoc 语义包。"""
        import json

        out = []
        if not os.path.isdir(self.root):
            return out
        for d in sorted(os.listdir(self.root)):
            mf_path = os.path.join(self.root, d, "_manifest.json")
            if not os.path.isfile(mf_path):
                continue
            try:
                with open(mf_path, encoding="utf-8") as fh:
                    mf = json.load(fh)
            except (OSError, ValueError):
                continue
            out.append(
                {
                    "id": d,  # 目录 slug（manifest/content API 的查找键）
                    "name": d,  # 展示名
                    "collection": mf.get("collection", f"adhoc:{d}"),
                    "dir": os.path.join(self.root, d),
                    "items": len(mf.get("items", [])),
                    "ok_items": sum(
                        1 for i in mf["items"] if i.get("status") == "ok"
                    ),
                    "updated_at": mf.get("updated_at"),
                }
            )
        return out

    def load_manifest(self, sname: str) -> Optional[dict]:
        import json

        path = os.path.join(self.adhoc_dir(sname), "_manifest.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------

    def process(
        self,
        paths: list[str],
        name: str,
        use_ocr: bool = True,
        recursive: bool = True,
        index: bool = True,
        progress: Optional[ProgressFn] = None,
    ) -> dict:
        """处理一组路径 → 转换 → 落盘 → 索引。返回摘要 dict（含增量信息）。"""
        t0 = time.time()
        sname = _slug(name, max_len=80)
        coll = self.collection_name(sname)
        adir = self.adhoc_dir(sname)
        os.makedirs(adir, exist_ok=True)

        files = scan_files(paths, recursive=recursive)
        old_manifest = self.load_manifest(sname)
        old_by_key = {e["item_key"]: e for e in (old_manifest or {}).get("items", [])}

        if progress:
            progress("collect", {"total": len(files), "collection": coll})

        entries: list[dict] = []
        all_chunks = []
        processed_keys: list[str] = []
        pending: list[AdhocFile] = []
        skipped = 0

        for i, f in enumerate(files):
            old = old_by_key.get(f.item_key)
            if (
                old is not None
                and old.get("status") == "ok"
                and old.get("source_mtime") == file_fingerprint(f.path).get("mtime")
                and os.path.isfile(os.path.join(old["dir"], "content.md"))
            ):
                entries.append(old)
                skipped += 1
                if progress:
                    progress(
                        "convert",
                        {"done": i + 1, "total": len(files),
                         "title": f.name[:40], "route": "skip"},
                    )
                continue

            if use_ocr and f.ext == ".pdf" and classify_pdf(f.path) == "ocr":
                pending.append(f)
                continue

            result = convert_attachment(f.path)
            self._finish(entries, all_chunks, processed_keys, adir, sname, f, result)
            if progress:
                progress(
                    "convert",
                    {"done": i + 1 - len(pending), "total": len(files),
                     "title": f.name[:40], "route": result.route},
                )

        ocr_used = False
        if pending:
            engine = self.ocr.get()
            ocr_used = True
            try:
                from .converter import pdf_ocr_to_markdown

                for j, f in enumerate(pending):
                    def prog(done, total, _f=f):
                        if progress:
                            progress(
                                "ocr",
                                {"title": _f.name[:40], "page": done, "pages": total},
                            )

                    result = pdf_ocr_to_markdown(f.path, engine, progress=prog)
                    self._finish(entries, all_chunks, processed_keys, adir, sname, f, result)
                    if progress:
                        progress(
                            "convert",
                            {"done": len(files) - len(pending) + j + 1,
                             "total": len(files), "title": f.name[:40],
                             "route": "ocr"},
                        )
            finally:
                self.ocr.release()

        manifest_path = os.path.join(adir, "_manifest.json")
        import json

        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"collection": coll, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "items": entries},
                fh, ensure_ascii=False, indent=2,
            )

        n_indexed = 0
        if index:
            current_keys = {e["item_key"] for e in entries}
            stale = {
                k for k in self.index.collection_item_keys(coll)
                if k not in current_keys
            } | set(processed_keys)
            if stale:
                self.index.delete_items(coll, sorted(stale))
            if all_chunks:
                n_indexed = self.index.add_chunks(all_chunks, self.embedder)

        summary = AdhocSummary(
            collection=coll,
            files=len(files),
            converted=len(files) - skipped,
            skipped=skipped,
            ocr_used=ocr_used,
            chunks_indexed=n_indexed,
            elapsed_sec=round(time.time() - t0, 1),
        )
        d = summary.to_dict()
        d["manifest"] = manifest_path
        if progress:
            progress("done", d)
        return d

    # ------------------------------------------------------------------

    def _finish(self, entries, all_chunks, processed_keys, adir, sname, f, result):
        import json

        from .indexer import chunk_markdown

        idir = os.path.join(adir, _slug(f.name))
        os.makedirs(idir, exist_ok=True)
        with open(os.path.join(idir, "content.md"), "w", encoding="utf-8") as fh:
            fh.write(result.markdown)

        meta = {
            "title": f.name,
            "source_file": f.path,
            "size_bytes": f.size,
            "conversion": {
                "route": result.route,
                "pages": result.pages,
                "page_labels": result.page_labels,
                "char_count": result.char_count,
                "notes": result.notes,
                "packaged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }
        with open(os.path.join(idir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        entry = {
            "item_key": f.item_key,
            "dir": idir,
            "title": f.name,
            "creator": "",
            "year": "",
            "route": result.route,
            "pages": result.pages,
            "chars": result.char_count,
            "status": "ok" if result.char_count > 0 else "empty",
            "notes": result.notes,
            "source_mtime": file_fingerprint(f.path).get("mtime"),
            "content_hash": md5_of_file(os.path.join(idir, "content.md")),
        }
        entries.append(entry)
        if result.char_count > 0:
            processed_keys.append(f.item_key)
            all_chunks.extend(
                chunk_markdown(
                    result.markdown,
                    item_key=f.item_key,
                    title=f.name,
                    collection=self.collection_name(sname),
                    source_file=f.path,
                )
            )

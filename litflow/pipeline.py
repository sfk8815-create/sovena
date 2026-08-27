"""litflow 流水线编排：分类 → 转换 → 语义包 → 向量索引。

「将某 Zotero 文库中的子分类做好 AI 调用准备」的落地实现：
    python -m litflow.pipeline 准备 "古琴研究" [--limit N] [--no-ocr]
    python -m litflow.pipeline 检索 "古琴音色" [--collection 古琴研究]
"""
from __future__ import annotations

import gc
import os
import time
from typing import Callable, Optional

from .converter import ConvertResult, convert_attachment
from .indexer import LMStudioEmbedder, VectorIndex, chunk_markdown
from .ocr_backends import resolve_backend
from .packager import Packager, file_fingerprint, md5_of_file
from .zotero_collector import LitRecord, ZoteroCollector

OCR_MODEL_DIR = os.environ.get(
    "LITFLOW_OCR_MODEL",
    "~/.lmstudio/models/LoJexLLM/Unlimited-OCR-MLX",
)
OCR_MODEL_DIR = os.path.expanduser(OCR_MODEL_DIR)

ProgressFn = Callable[[str, dict], None]


class OCREngineHolder:
    """OCR 引擎懒加载/卸载（避免常驻内存，配合调度器）。

    后端二选一（详见 ocr_backends 模块文档）：
      - mlx ：本机 ocr_port（Apple Silicon）
      - http：OpenAI 兼容服务（llama-server / LM Studio 等跑 GGUF，全平台）
    """

    def __init__(self, model_dir: str = OCR_MODEL_DIR):
        self.model_dir = model_dir
        self.backend = resolve_backend()
        self._engine = None

    def get(self):
        if self._engine is None:
            if self.backend == "http":
                from .ocr_backends import OpenAICompatOCREngine

                self._engine = OpenAICompatOCREngine()
            else:
                from ocr_port import UnlimitedOCRInference

                self._engine = UnlimitedOCRInference(self.model_dir).load()
        return self._engine

    def release(self):
        if self._engine is not None:
            self._engine = None
            gc.collect()


class Pipeline:
    def __init__(
        self,
        collector: Optional[ZoteroCollector] = None,
        packager: Optional[Packager] = None,
        index: Optional[VectorIndex] = None,
        embedder: Optional[LMStudioEmbedder] = None,
        ocr: Optional[OCREngineHolder] = None,
    ):
        self.collector = collector or ZoteroCollector()
        self.packager = packager or Packager()
        self.index = index or VectorIndex()
        self.embedder = embedder or LMStudioEmbedder()
        self.ocr = ocr or OCREngineHolder()

    # ------------------------------------------------------------------

    def prepare(
        self,
        collection: str,
        recursive: bool = True,
        limit: Optional[int] = None,
        use_ocr: bool = True,
        progress: Optional[ProgressFn] = None,
        rebuild: bool = False,
    ) -> dict:
        """准备一个分类：采集 → 转换 → 落盘 → 索引。返回摘要。

        增量机制：与该分类上次 manifest 对比，Zotero 条目 version 未变
        且 content.md 完好的条目直接跳过；索引按条目级增删，不全量重建。
        rebuild=True 时全量重来。
        """
        t0 = time.time()
        records = self.collector.collect(collection, recursive=recursive)
        coll = self.collector.resolve_collection(collection)
        coll_name = coll["name"]
        if limit:
            records = records[:limit]

        old_manifest = self.packager.load_manifest(coll_name)
        old_by_key = {}
        if old_manifest:
            old_by_key = {e["item_key"]: e for e in old_manifest.get("items", [])}

        if progress:
            progress("collect", {"total": len(records), "collection": coll_name})

        entries: list[dict] = []      # [(原序号, entry)]
        processed_keys: list[str] = []  # 本次实际转换的条目
        all_chunks = []
        ocr_used = False
        pending = []
        skipped = 0

        for i, rec in enumerate(records):
            att = self._pick_attachment(rec)
            source = (att.path or att.url) if att else None

            # ---- 增量跳过 ----
            old = old_by_key.get(rec.item_key)
            if (
                not rebuild
                and old is not None
                and old.get("status") == "ok"
                and int(old.get("zotero_version") or 0) == rec.version
                and (source is None or old.get("source_mtime") == file_fingerprint(source).get("mtime"))
                and os.path.isfile(os.path.join(old["dir"], "content.md"))
            ):
                entries.append((i, old))
                skipped += 1
                if progress:
                    progress(
                        "convert",
                        {"done": i + 1, "total": len(records),
                         "title": rec.title[:40], "route": "skip"},
                    )
                continue

            result = ConvertResult(markdown="", route="empty")
            if att:
                needs_ocr = (
                    att.is_pdf
                    and use_ocr
                    and os.path.isfile(att.path or "")
                    and _pdf_needs_ocr(att.path)
                )
                if needs_ocr:
                    pending.append((i, rec, att))
                    continue
                result = convert_attachment(att.path, att.content_type, att.url)
            entry, chunks = self._package_and_chunk(coll_name, rec, result, source)
            entries.append((i, entry))
            if result.char_count > 0:
                processed_keys.append(rec.item_key)
            all_chunks.extend(chunks)
            if progress:
                progress(
                    "convert",
                    {"done": i + 1 - len(pending), "total": len(records),
                     "title": rec.title[:40], "route": result.route},
                )

        # OCR 通道：统一在最后处理（引擎一次性加载，转换完即释放）
        if pending:
            engine = self.ocr.get()
            ocr_used = True
            try:
                for j, (i, rec, att) in enumerate(pending):
                    from .converter import pdf_ocr_to_markdown

                    def prog(done, total, _rec=rec):
                        if progress:
                            progress(
                                "ocr",
                                {"title": _rec.title[:40], "page": done, "pages": total},
                            )

                    result = pdf_ocr_to_markdown(att.path, engine, progress=prog)
                    entry, chunks = self._package_and_chunk(
                        coll_name, rec, result, att.path
                    )
                    entries.append((i, entry))
                    if result.char_count > 0:
                        processed_keys.append(rec.item_key)
                    all_chunks.extend(chunks)
                    if progress:
                        progress(
                            "convert",
                            {"done": len(records) - len(pending) + j + 1,
                             "total": len(records), "title": rec.title[:40],
                             "route": "ocr"},
                        )
            finally:
                self.ocr.release()

        entries.sort(key=lambda e: e[0])
        entries = [e[1] for e in entries]

        manifest_path = self.packager.write_manifest(coll_name, entries)

        # ---- 索引增量 ----
        n_indexed = 0
        if rebuild:
            self.index.reset_collection(coll_name)
            if all_chunks:
                n_indexed = self.index.add_chunks(all_chunks, self.embedder)
        else:
            # 删除：已不在分类中 or 本次重新转换的条目
            current_keys = {e["item_key"] for e in entries}
            stale = {
                k for k in self.index.collection_item_keys(coll_name)
                if k not in current_keys
            } | set(processed_keys)
            if stale:
                self.index.delete_items(coll_name, sorted(stale))
            if all_chunks:
                n_indexed = self.index.add_chunks(all_chunks, self.embedder)

        summary = {
            "collection": coll_name,
            "records": len(records),
            "skipped": skipped,
            "converted": len(records) - skipped,
            "ocr_used": ocr_used,
            "manifest": manifest_path,
            "chunks_indexed": n_indexed,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        if progress:
            progress("done", summary)
        return summary

    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        collection: Optional[str] = None,
        top_k: int = 8,
    ) -> list[dict]:
        return self.index.search(query, self.embedder, collection=collection, top_k=top_k)

    # ------------------------------------------------------------------

    def _pick_attachment(self, rec: LitRecord):
        """优先 PDF 快照，其次网页快照，其次链接 URL。"""
        if not rec.attachments:
            return None
        pdfs = [a for a in rec.attachments if a.is_pdf and a.exists]
        if pdfs:
            return pdfs[0]
        snaps = [a for a in rec.attachments if a.is_snapshot and a.path]
        if snaps:
            return snaps[0]
        urls = [a for a in rec.attachments if a.url]
        if urls:
            return urls[0]
        return rec.attachments[0]

    def _package_and_chunk(self, coll_name, rec, result, source):
        idir = self.packager.write_item(coll_name, rec, result, source_path=source)
        entry = {
            "item_key": rec.item_key,
            "dir": idir,
            "title": rec.title,
            "creator": rec.creator_summary(),
            "year": rec.year,
            "route": result.route,
            "pages": result.pages,
            "chars": result.char_count,
            "status": "ok" if result.char_count > 0 else "empty",
            "notes": result.notes,
            "zotero_version": rec.version,
            "source_mtime": file_fingerprint(source).get("mtime"),
            "content_hash": md5_of_file(os.path.join(idir, "content.md")),
        }
        chunks = chunk_markdown(
            result.markdown,
            item_key=rec.item_key,
            title=rec.title,
            creator=rec.creator_summary(),
            year=rec.year,
            collection=coll_name,
            source_file=source,
        ) if result.char_count > 0 else []
        return entry, chunks


def _pdf_needs_ocr(path: Optional[str]) -> bool:
    from .converter import classify_pdf

    return bool(path) and os.path.isfile(path) and classify_pdf(path) == "ocr"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_progress(stage, info):
    if stage == "convert":
        print(f"  [{info['done']}/{info['total']}] {info.get('route'):6s} {info.get('title', '')}")
    elif stage == "ocr":
        print(f"    OCR {info['title']}: {info['page']}/{info['pages']} 页", flush=True)
    elif stage == "done":
        print("完成:", info)
    elif stage == "collect":
        print(f"采集 {info['collection']}: {info['total']} 条")


def main(argv: Optional[list[str]] = None):
    import argparse

    ap = argparse.ArgumentParser(description="litflow 文献流水线")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("准备", help="将分类做好 AI 调用准备（默认增量）")
    p1.add_argument("collection")
    p1.add_argument("--limit", type=int, default=None)
    p1.add_argument("--no-ocr", action="store_true")
    p1.add_argument("--rebuild", action="store_true", help="忽略增量，全量重建")

    p2 = sub.add_parser("检索", help="自然语言文献检索")
    p2.add_argument("query")
    p2.add_argument("--collection", default=None)
    p2.add_argument("--top-k", type=int, default=8)

    args = ap.parse_args(argv)
    pipe = Pipeline()

    if args.cmd == "准备":
        pipe.prepare(args.collection, limit=args.limit, use_ocr=not args.no_ocr,
                     progress=_cli_progress, rebuild=args.rebuild)
    else:
        hits = pipe.search(args.query, collection=args.collection, top_k=args.top_k)
        for h in hits:
            print(f"\n[{h['collection']}] {h['creator']} {h['year']} 《{h['title']}》 p.{h['page']}")
            print("   ", h["text"][:120].replace("\n", " "))


if __name__ == "__main__":
    main()

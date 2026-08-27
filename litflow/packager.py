"""litflow 语义文献包落盘。

目录结构（根目录默认 ~/litflow_data，可用 LITFLOW_ROOT 覆盖）：
  <根>/<分类名>/
    _manifest.json                 # 分类级清单（条目 ↔ 包目录、状态）
    <作者>_<年份>_<标题slug>/
      meta.json                    # Zotero 元数据 + 转换统计
      content.md                   # AI 友好 markdown（含页码标注）
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from typing import Optional

from .converter import ConvertResult
from .zotero_collector import LitRecord

DEFAULT_ROOT = os.environ.get("LITFLOW_ROOT") or os.path.expanduser(
    "~/litflow_data"
)


def file_fingerprint(path: Optional[str]) -> dict:
    """源文件指纹（mtime + size），用于增量检测。"""
    if not path or not os.path.isfile(path):
        return {"mtime": None, "size": None}
    st = os.stat(path)
    return {"mtime": int(st.st_mtime), "size": st.st_size}


def md5_of_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _slug(text: str, max_len: int = 60) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\\/:*?\"<>|]+", " ", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len].strip("_") or "untitled"


def item_dir_name(rec: LitRecord) -> str:
    creator = rec.creator_summary().replace(" ", "")[:20] or "佚名"
    year = rec.year or "无年份"
    return f"{creator}_{year}_{_slug(rec.title)}"


class Packager:
    def __init__(self, root: str = DEFAULT_ROOT):
        self.root = root

    def collection_dir(self, collection_name: str) -> str:
        return os.path.join(self.root, _slug(collection_name, max_len=80))

    def write_item(
        self,
        collection_name: str,
        rec: LitRecord,
        result: ConvertResult,
        source_path: Optional[str] = None,
    ) -> str:
        """写一个条目的语义包，返回包目录路径。幂等（覆盖写）。"""
        cdir = self.collection_dir(collection_name)
        idir = os.path.join(cdir, item_dir_name(rec))
        os.makedirs(idir, exist_ok=True)

        with open(os.path.join(idir, "content.md"), "w", encoding="utf-8") as f:
            f.write(result.markdown)

        meta = {
            "zotero_key": rec.item_key,
            "title": rec.title,
            "item_type": rec.item_type,
            "creators": [
                c.get("lastName") or c.get("name", "")
                for c in rec.creators
            ],
            "creator_summary": rec.creator_summary(),
            "year": rec.year,
            "publication": rec.publication,
            "abstract": rec.abstract,
            "tags": rec.tags,
            "url": rec.url,
            "doi": rec.doi,
            "collections": rec.collections,
            "source_file": source_path,
            "conversion": {
                "route": result.route,
                "pages": result.pages,
                "page_labels": result.page_labels,
                "char_count": result.char_count,
                "notes": result.notes,
                "packaged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }
        with open(os.path.join(idir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return idir

    def write_manifest(
        self, collection_name: str, entries: list[dict]
    ) -> str:
        """分类清单：[{item_key, dir, title, route, pages, chars, status}]"""
        cdir = self.collection_dir(collection_name)
        os.makedirs(cdir, exist_ok=True)
        manifest = {
            "collection": collection_name,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "items": entries,
        }
        path = os.path.join(cdir, "_manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return path

    def load_manifest(self, collection_name: str) -> Optional[dict]:
        path = os.path.join(self.collection_dir(collection_name), "_manifest.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

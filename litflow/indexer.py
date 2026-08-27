"""litflow 向量索引与自然语言检索（LanceDB + LM Studio embedding）。

- 嵌入模型：LM Studio OpenAI 兼容 API（默认 qwen3-embedding-4b，2560 维）
- 分块：按页码标注切块（页边界优先，块内保留页码上下文）
- 检索：cos 相似度 + 元数据过滤（分类/标题/年份）
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx
import lancedb
import numpy as np

PAGE_MARK = re.compile(r"\*\*\[p\.([^\]]+)\]\*\*")

DEFAULT_LMSTUDIO = os.environ.get("LITFLOW_LMSTUDIO", "http://localhost:1234/v1")
DEFAULT_EMBED_MODEL = os.environ.get(
    "LITFLOW_EMBED_MODEL", "text-embedding-qwen3-embedding-4b"
)
# 向量库默认跟随 LITFLOW_ROOT（仅设置 LITFLOW_ROOT 也可整体迁移）
DEFAULT_DB_PATH = os.environ.get("LITFLOW_LANCEDB") or os.path.join(
    os.environ.get(
        "LITFLOW_ROOT", "/Volumes/macstudio-work/synology_drive/zotero_AI"
    ),
    "_lancedb",
)

CHUNK_TARGET = 900   # 目标块字符数（中文友好）
CHUNK_OVERLAP = 120


def _sql_escape(s: str) -> str:
    """LanceDB SQL 过滤字符串转义（单引号）。"""
    return s.replace("'", "''")


# ---------------------------------------------------------------------------
# 分块
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    page: Optional[str]
    item_key: str
    title: str
    creator: str
    year: Optional[str]
    collection: str
    source_file: Optional[str]


def chunk_markdown(
    md: str,
    item_key: str,
    title: str,
    creator: str = "",
    year: Optional[str] = None,
    collection: str = "",
    source_file: Optional[str] = None,
) -> list[Chunk]:
    """按页标记切分，长页再滑窗切分。返回带页码元数据的块。"""
    # 依页标记切分
    pieces: list[tuple[Optional[str], str]] = []
    last_end, last_label = 0, None
    for m in PAGE_MARK.finditer(md):
        if m.start() > last_end and md[last_end:m.start()].strip():
            pieces.append((last_label, md[last_end:m.start()]))
        last_label = m.group(1)
        last_end = m.end()
    if md[last_end:].strip():
        pieces.append((last_label, md[last_end:]))

    def _mk(text: str, label: Optional[str]) -> Chunk:
        return Chunk(
            text=text,
            page=label,
            item_key=item_key,
            title=title,
            creator=creator,
            year=year,
            collection=collection,
            source_file=source_file,
        )

    chunks: list[Chunk] = []
    for label, body in pieces:
        body = body.strip()
        if not body:
            continue
        # 长页滑窗
        if len(body) <= CHUNK_TARGET * 1.5:
            chunks.append(_mk(body, label))
            continue
        step = CHUNK_TARGET - CHUNK_OVERLAP
        for start in range(0, len(body), step):
            seg = body[start : start + CHUNK_TARGET]
            if not seg.strip():
                continue
            chunks.append(_mk(seg, label))
    return chunks


# ---------------------------------------------------------------------------
# 嵌入客户端
# ---------------------------------------------------------------------------

class LMStudioEmbedder:
    def __init__(
        self,
        base_url: str = DEFAULT_LMSTUDIO,
        model: str = DEFAULT_EMBED_MODEL,
        batch: int = 32,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch = batch
        self._client = httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch):
            r = self._client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": texts[i : i + self.batch]},
            )
            r.raise_for_status()
            data = r.json()["data"]
            data.sort(key=lambda e: e["index"])
            out.extend(e["embedding"] for e in data)
        arr = np.asarray(out, dtype=np.float32)
        arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------

SCHEMA = [
    "vector",         # dims 由首次写入决定
    "text",
    "page",
    "item_key",
    "title",
    "creator",
    "year",
    "collection",
    "source_file",
]


class VectorIndex:
    def __init__(self, db_path: str = DEFAULT_DB_PATH, table: str = "chunks"):
        self.db = lancedb.connect(db_path)
        self.table_name = table
        self._table = None

    def _get_table(self):
        # 每次重新打开：其他进程/实例可能已 drop 并重建该表，
        # 缓存旧句柄会指向已删除的数据文件
        if self.table_name in self.db.table_names():
            self._table = self.db.open_table(self.table_name)
        else:
            self._table = None
        return self._table

    def reset_collection(self, collection: str) -> None:
        """重建前清空某分类的块。"""
        import pyarrow as pa
        import pyarrow.compute as pc

        t = self._get_table()
        if t is None:
            return
        tbl = t.to_arrow()
        keep = tbl.filter(pc.invert(pc.equal(tbl.column("collection"), collection)))
        self.db.drop_table(self.table_name)
        if len(keep) > 0:
            self._table = self.db.create_table(self.table_name, keep)
        else:
            self._table = None

    def delete_items(self, collection: str, item_keys: list[str]) -> int:
        """增量索引：删除某些条目在该分类下的块。返回删除的条目数。"""
        t = self._get_table()
        if t is None or not item_keys:
            return 0
        keys = ", ".join(f"'{k}'" for k in item_keys)  # item_key 为 8 位字母数字
        t.delete(f"collection = '{_sql_escape(collection)}' AND item_key IN ({keys})")
        return len(item_keys)

    def collection_item_keys(self, collection: str) -> set[str]:
        """某分类当前已索引的 item_key 集合。"""
        t = self._get_table()
        if t is None:
            return set()
        import pyarrow.compute as pc

        tbl = t.to_arrow()
        col = tbl.column("collection")
        sel = tbl.filter(pc.equal(col, collection))
        return set(sel.column("item_key").to_pylist())

    def add_chunks(self, chunks: list[Chunk], embedder: LMStudioEmbedder) -> int:
        if not chunks:
            return 0
        vecs = embedder.embed([c.text for c in chunks])
        import pyarrow as pa

        data = pa.table(
            {
                "vector": pa.array(vecs.tolist(), type=pa.list_(pa.float32(), vecs.shape[1])),
                "text": [c.text for c in chunks],
                "page": [c.page or "" for c in chunks],
                "item_key": [c.item_key for c in chunks],
                "title": [c.title for c in chunks],
                "creator": [c.creator for c in chunks],
                "year": [c.year or "" for c in chunks],
                "collection": [c.collection for c in chunks],
                "source_file": [c.source_file or "" for c in chunks],
            }
        )
        if self._get_table() is None:
            self._table = self.db.create_table(self.table_name, data)
        else:
            self._table.add(data)
        return len(chunks)

    def search(
        self,
        query: str,
        embedder: LMStudioEmbedder,
        collection: Optional[str] = None,
        top_k: int = 8,
    ) -> list[dict]:
        t = self._get_table()
        if t is None:
            return []
        qv = embedder.embed([query])[0].tolist()
        q = (
            t.search(qv)
            .metric("cosine")
            .limit(top_k * (4 if collection else 1))
            .select(["text", "page", "item_key", "title", "creator", "year", "collection", "source_file"])
        )
        if collection:
            q = q.where(f"collection = '{_sql_escape(collection)}'")
        arrow = q.to_arrow()
        out = []
        for row in arrow.to_pylist():
            row.pop("_distance", None)
            out.append(row)
        return out[:top_k]

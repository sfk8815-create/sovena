"""litflow Zotero 采集器。

通过 Zotero 本地 API（http://localhost:23119/api）读取文库分类与条目，
并解析 4 路附件为本地可读文件/URL：
  1. `attachments:相对路径` → Zotero 链接附件基目录（baseAttachmentPath）
  2. `storage:文件名`       → Zotero 数据目录 storage/<附件key>/文件名
  3. 绝对路径               → 原样使用
  4. linked_url             → 保留 URL（网页快照/在线资源）

数据目录与链接基目录自动从 Zotero prefs.js 读取，可被构造参数覆盖。
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

DEFAULT_API = os.environ.get("LITFLOW_ZOTERO_API", "http://localhost:23119/api")

PDF_CONTENT_TYPES = {"application/pdf"}
SNAPSHOT_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}


@dataclass
class Attachment:
    """一条附件的解析结果。"""
    key: str
    content_type: Optional[str]
    filename: str
    link_mode: str  # imported_file | linked_file | linked_url | embedded_image
    path: Optional[str] = None   # 本地文件路径（存在则非 None）
    url: Optional[str] = None    # linked_url 或 path 元数据中的链接
    exists: bool = False

    @property
    def is_pdf(self) -> bool:
        return self.path is not None and self.content_type in PDF_CONTENT_TYPES

    @property
    def is_snapshot(self) -> bool:
        return self.content_type in SNAPSHOT_CONTENT_TYPES


@dataclass
class LitRecord:
    """一条文献条目（含附件）。"""
    item_key: str
    item_type: str
    title: str
    version: int = 0  # Zotero 条目版本号（每次修改递增，用于增量检测）
    creators: list[dict] = field(default_factory=list)
    year: Optional[str] = None
    publication: Optional[str] = None
    abstract: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    url: Optional[str] = None
    doi: Optional[str] = None
    collections: list[str] = field(default_factory=list)  # 所在分类 key
    attachments: list[Attachment] = field(default_factory=list)

    def creator_summary(self) -> str:
        if not self.creators:
            return ""
        first = self.creators[0].get("lastName") or self.creators[0].get("name") or ""
        n = len(self.creators)
        if n == 1:
            return first
        return f"{first} 等({n})"


class ZoteroPrefs:
    """从 Zotero prefs.js 读取 dataDir / baseAttachmentPath。"""

    _PREFS_GLOB = os.path.expanduser(
        "~/Library/Application Support/Zotero/Profiles/*/prefs.js"
    )

    @classmethod
    def read(cls) -> dict:
        out = {}
        for p in sorted(glob.glob(cls._PREFS_GLOB)):
            try:
                text = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for key, attr in (
                ("extensions.zotero.dataDir", "data_dir"),
                ("extensions.zotero.baseAttachmentPath", "linked_base"),
            ):
                m = re.search(
                    rf'user_pref\("{re.escape(key)}",\s*"([^"]+)"\)', text
                )
                if m and attr not in out:
                    out[attr] = m.group(1)
        return out


class ZoteroCollector:
    def __init__(
        self,
        api_url: str = DEFAULT_API,
        user_id: int | str = 0,
        data_dir: Optional[str] = None,
        linked_base: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_url = api_url.rstrip("/")
        self.user_id = user_id
        prefs = ZoteroPrefs.read()
        self.data_dir = data_dir or prefs.get("data_dir") or os.path.expanduser("~/Zotero")
        self.linked_base = linked_base or prefs.get("linked_base") or ""
        self._client = httpx.Client(base_url=self.api_url, timeout=timeout)

    # ---------------- 底层 API ----------------

    def _get(self, path: str, **params) -> list | dict:
        r = self._client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def _paginated(self, path: str, limit: int = 100, **params) -> list:
        items: list = []
        start = 0
        while True:
            batch = self._get(path, start=start, limit=limit, **params)
            if not batch:
                break
            items.extend(batch)
            if len(batch) < limit:
                break
            start += limit
        return items

    # ---------------- 分类 ----------------

    def collections(self) -> list[dict]:
        """全部分类：[{key, name, parent, num_items, num_collections}]"""
        raw = self._paginated(f"users/{self.user_id}/collections", format="json")
        out = []
        for c in raw:
            d = c["data"]
            out.append(
                {
                    "key": d["key"],
                    "name": d["name"],
                    "parent": d.get("parentCollection") or None,
                    "num_items": c["meta"]["numItems"],
                    "num_collections": c["meta"]["numCollections"],
                }
            )
        return out

    def collection_tree(self) -> dict:
        """分类树 {name: {key, num_items, children: {…}}}"""
        cols = self.collections()
        nodes = {
            c["key"]: {"key": c["key"], "num_items": c["num_items"], "children": {}}
            for c in cols
        }
        roots: dict = {}
        for c in cols:
            node = nodes[c["key"]]
            parent = c["parent"]
            if parent in nodes:
                nodes[parent]["children"][c["name"]] = node
            else:
                roots[c["name"]] = node
        return roots

    def resolve_collection(self, name_or_key: str) -> Optional[dict]:
        """按 key 或名称（精确→包含）解析分类。"""
        cols = self.collections()
        for c in cols:
            if c["key"] == name_or_key or c["name"] == name_or_key:
                return c
        for c in cols:
            if name_or_key in c["name"]:
                return c
        return None

    def subcollection_keys(self, coll_key: str, recursive: bool = True) -> list[str]:
        cols = self.collections()
        children = {c["key"] for c in cols if c["parent"] == coll_key}
        if not recursive:
            return sorted(children)
        all_keys = set(children)
        frontier = list(children)
        while frontier:
            cur = frontier.pop()
            for c in cols:
                if c["parent"] == cur and c["key"] not in all_keys:
                    all_keys.add(c["key"])
                    frontier.append(c["key"])
        return sorted(all_keys)

    # ---------------- 条目与附件 ----------------

    def collection_items(
        self, coll_key: str, recursive: bool = True
    ) -> tuple[list[LitRecord], dict[str, list[Attachment]]]:
        """返回 (条目列表, 父条目key → 附件列表)。附件来自同一分页响应，无需逐条目请求。"""
        keys = [coll_key] + (self.subcollection_keys(coll_key, recursive) if recursive else [])
        records: dict[str, LitRecord] = {}
        attachments: dict[str, list[Attachment]] = {}
        for ck in keys:
            raw = self._paginated(f"users/{self.user_id}/collections/{ck}/items", format="json")
            for it in raw:
                d = it["data"]
                if d["itemType"] == "attachment":
                    parent = d.get("parentItem")
                    if parent:
                        attachments.setdefault(parent, []).append(
                            self._resolve_attachment(it["key"], d)
                        )
                    continue
                if d["itemType"] in ("note", "annotation"):
                    continue
                rec = records.get(d["key"])
                if rec is None:
                    rec = self._to_record(d, version=it.get("version", 0))
                    records[d["key"]] = rec
                rec.collections.append(ck)
        return list(records.values()), attachments

    def _to_record(self, d: dict, version: int = 0) -> LitRecord:
        rec = LitRecord(
            item_key=d["key"],
            item_type=d["itemType"],
            title=d.get("title") or d.get("name") or "(无标题)",
            version=version,
            creators=d.get("creators", []),
            year=(d.get("date") or "")[:4] or None,
            publication=d.get("publicationTitle") or d.get("publisher"),
            abstract=d.get("abstractNote"),
            tags=[t.get("tag", "") for t in d.get("tags", [])],
            url=d.get("url"),
            doi=d.get("DOI"),
        )
        return rec

    def item_attachments(self, item_key: str) -> list[Attachment]:
        raw = self._get(f"users/{self.user_id}/items/{item_key}/children", format="json")
        out = []
        for it in raw:
            d = it["data"]
            if d["itemType"] != "attachment":
                continue
            out.append(self._resolve_attachment(it["key"], d))
        return out

    # ---------------- 搜索 / 条目详情 / 标注 ----------------

    def search_items(
        self,
        q: Optional[str] = None,
        title: Optional[str] = None,
        creator: Optional[str] = None,
        year: Optional[str] = None,
        tag: Optional[str] = None,
        item_type: Optional[str] = None,
        limit: int = 30,
    ) -> list[LitRecord]:
        """多维元数据搜索（q 走 Zotero 服务端搜索，其余客户端过滤）。"""
        raw = self._paginated(
            f"users/{self.user_id}/items",
            format="json",
            q=q or "",
            itemType=item_type or "-attachment -note -annotation",
        )
        out: list[LitRecord] = []
        for it in raw:
            d = it["data"]
            if title and title.lower() not in (d.get("title") or "").lower():
                continue
            if year and not (d.get("date") or "").startswith(str(year)):
                continue
            if creator:
                names = " ".join(
                    (c.get("lastName") or "") + (c.get("firstName") or "") + (c.get("name") or "")
                    for c in d.get("creators", [])
                )
                if creator.lower() not in names.lower():
                    continue
            if tag and tag not in [t.get("tag", "") for t in d.get("tags", [])]:
                continue
            rec = self._to_record(d, version=it.get("version", 0))
            rec.attachments = self.item_attachments(rec.item_key)
            out.append(rec)
            if len(out) >= limit:
                break
        return out

    def get_item(self, item_key: str) -> Optional[LitRecord]:
        """按 key 获取单条条目（含附件）。"""
        r = self._client.get(
            f"users/{self.user_id}/items/{item_key}", params={"format": "json"}
        )
        if r.status_code != 200:
            return None
        it = r.json()
        rec = self._to_record(it["data"], version=it.get("version", 0))
        rec.attachments = self.item_attachments(item_key)
        return rec

    def item_annotations(self, item_key: str) -> list[dict]:
        """读取条目 PDF 附件上的标注（高亮/批注等，Zotero annotation 子条目）。"""
        atts = self.item_attachments(item_key)
        out: list[dict] = []
        for att in atts:
            raw = self._get(
                f"users/{self.user_id}/items/{att.key}/children", format="json"
            )
            for it in raw:
                d = it["data"]
                if d.get("itemType") != "annotation":
                    continue
                out.append(
                    {
                        "attachment": att.filename,
                        "type": d.get("annotationType", ""),
                        "text": d.get("annotationText") or "",
                        "comment": d.get("annotationComment") or "",
                        "color": d.get("annotationColor") or "",
                        "page_label": d.get("annotationPageLabel") or "",
                        "tags": [t.get("tag", "") for t in d.get("tags", [])],
                        "modified": d.get("dateModified", ""),
                    }
                )
        out.sort(key=lambda a: a["modified"], reverse=True)
        return out

    def item_notes(self, item_key: str) -> list[dict]:
        """读取条目下的笔记。"""
        raw = self._get(f"users/{self.user_id}/items/{item_key}/children", format="json")
        out = []
        for it in raw:
            d = it["data"]
            if d.get("itemType") == "note":
                out.append(
                    {
                        "key": it["key"],
                        "html": d.get("note", ""),
                        "tags": [t.get("tag", "") for t in d.get("tags", [])],
                        "modified": d.get("dateModified", ""),
                    }
                )
        return out

    def _resolve_attachment(self, att_key: str, d: dict) -> Attachment:
        path = d.get("path") or ""
        filename = d.get("filename") or os.path.basename(path.replace(":", "/"))
        link_mode = d.get("linkMode", "")
        att = Attachment(
            key=att_key,
            content_type=d.get("contentType"),
            filename=filename,
            link_mode=link_mode,
            url=d.get("url"),
        )
        local: Optional[str] = None
        if path.startswith("attachments:"):
            rel = path[len("attachments:"):]
            if self.linked_base:
                local = os.path.join(self.linked_base, rel)
        elif path.startswith("storage:"):
            fname = path[len("storage:"):]
            local = os.path.join(self.data_dir, "storage", att_key, fname)
        elif os.path.isabs(path):
            local = path
        # linked_url：仅保留 url
        if local:
            local = os.path.normpath(local)
            att.path = local
            att.exists = os.path.isfile(local)
        return att

    # ---------------- 顶层入口 ----------------

    def collect(self, collection: str, recursive: bool = True) -> list[LitRecord]:
        """把某分类（及子分类）中的文献条目连同附件收集为 LitRecord 列表。"""
        coll = self.resolve_collection(collection)
        if coll is None:
            raise ValueError(f"未找到分类: {collection}")
        records, attachments = self.collection_items(coll["key"], recursive=recursive)
        for rec in records:
            rec.attachments = attachments.get(rec.item_key, [])
        return records

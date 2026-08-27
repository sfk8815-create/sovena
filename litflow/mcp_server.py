"""litflow MCP 服务：把文献流能力以 MCP 工具形式暴露给 AI 客户端。

与 Web UI 同进程同端口：
    python -m litflow.server   （或 uv run litflow serve）
    → MCP 端点 http://localhost:8765/mcp（远程经 tailscale 用机器 IP）

工具一览（对标并超越 zotero-mcp）：
    ---- Zotero 读取（对标 search/collection/content 类）----
    zotero_collections       分类树 + 准备状态
    zotero_search            多维元数据搜索（q/标题/作者/年份/标签）
    zotero_item              条目详情（元数据+附件+笔记）
    zotero_annotations       PDF 标注（高亮/批注，按颜色/类型）
    ---- litflow 核心（独有能力）----
    litflow_prepare          一键「将子分类做好 AI 调用准备」（默认增量）
    litflow_job_status       任务进度/日志
    litflow_jobs             任务列表
    litflow_cancel_job       取消任务
    litflow_search           自然语言语义检索（跨语言、带原书页码）
    litflow_find_similar     按条目找语义相似文献
    litflow_manifest         分类文献包清单
    litflow_read_item        读取文献 markdown 全文
    ---- adhoc 临时资料（独有）----
    litflow_adhoc_process    任意文件/文件夹 → OCR+向量索引
    litflow_adhoc_list       已有资料包列表
    ---- 运维 ----
    litflow_doctor           连通性诊断（Zotero/LM Studio/目录权限）
    litflow_system_status    系统资源状态

注：Zotero 本地 API 只读，故不提供写条目工具（与进程内插件方案的差异）。
"""
from __future__ import annotations

import os
import time

import httpx
from fastmcp import FastMCP

from . import web as webmod
from .jobs import get_manager, system_status
from .packager import Packager
from .pipeline import Pipeline
from .zotero_collector import ZoteroCollector

mcp = FastMCP(
    "litflow",
    instructions=(
        "litflow：学科学术文献流服务（人文/社科/理工/医学等通用）。工作流程：① zotero_collections / zotero_search "
        "找到 Zotero 分类或条目；② litflow_prepare 把分类转换为语义文献包并建向量索引"
        "（默认增量，大分类耗时长，可传 wait=False 后台执行并用 litflow_job_status 轮询）；"
        "③ litflow_search 做自然语言检索（支持跨语言，命中带原书页码 page 字段）；"
        "④ litflow_read_item 读取全文。非 Zotero 的本地资料（任意文件夹/文件）用 "
        "litflow_adhoc_process 处理后同样可检索。引用时注意 page 是原书页码非物理页序。"
    ),
)

_packager = Packager()
_collector = ZoteroCollector()
_pipe = Pipeline()

DEFAULT_HOST = os.environ.get("LITFLOW_HOST", "0.0.0.0")   # 0.0.0.0 供 tailscale 远程访问
DEFAULT_PORT = int(os.environ.get("LITFLOW_PORT", "8765"))


# ---------------------------------------------------------------------------
# Zotero 读取类
# ---------------------------------------------------------------------------

@mcp.tool
def zotero_collections() -> str:
    """列出 Zotero 文库全部分类（层级缩进）及各分类语义包准备状态。"""
    colls = _collector.collections()
    by_key = {c["key"]: c for c in colls}
    lines = []

    def walk(c: dict, depth: int):
        mf = _packager.load_manifest(c["name"])
        if mf:
            ok = sum(1 for i in mf["items"] if i["status"] == "ok")
            state = f"已准备（{ok}/{len(mf['items'])} 条就绪，更新于 {mf.get('updated_at')}）"
        else:
            state = "未准备"
        lines.append(f"{'  ' * depth}{c['name']}（{c['num_items']} 条）：{state}")
        for ch in colls:
            if ch["parent"] == c["key"]:
                walk(ch, depth + 1)

    for c in colls:
        if c["parent"] not in by_key:
            walk(c, 0)
    return "\n".join(lines) or "（Zotero 中没有分类）"


@mcp.tool
def zotero_search(
    q: str | None = None,
    title: str | None = None,
    creator: str | None = None,
    year: str | None = None,
    tag: str | None = None,
    limit: int = 20,
) -> str:
    """按元数据搜索 Zotero 条目（q 为全文关键词，其余为精确过滤）。

    返回条目 key、标题、作者、年份、附件情况。拿到 item_key 后可用
    zotero_item 看详情，或用 litflow_read_item 读已准备的全文。
    """
    recs = _collector.search_items(
        q=q, title=title, creator=creator, year=year, tag=tag, limit=limit
    )
    if not recs:
        return "无匹配条目。"
    out = []
    for r in recs:
        atts = [a.filename for a in r.attachments if a.exists]
        att_desc = f"附件: {', '.join(atts[:3])}" if atts else "无本地附件"
        out.append(
            f"[{r.item_key}] {r.title}（{r.creator_summary()} {r.year or ''}）{att_desc}"
        )
    return "\n".join(out)


@mcp.tool
def zotero_item(item_key: str) -> str:
    """获取单条 Zotero 条目的完整信息：元数据、附件、笔记。"""
    rec = _collector.get_item(item_key)
    if rec is None:
        return f"条目 {item_key} 不存在。"
    parts = [
        f"标题：{rec.title}",
        f"类型：{rec.item_type}  作者：{rec.creator_summary()}  年份：{rec.year or '-'}",
    ]
    if rec.publication:
        parts.append(f"出版物：{rec.publication}")
    if rec.abstract:
        parts.append(f"摘要：{rec.abstract[:600]}")
    if rec.tags:
        parts.append(f"标签：{'、'.join(rec.tags)}")
    if rec.url or rec.doi:
        parts.append(f"链接：{rec.url or ''} {('DOI: ' + rec.doi) if rec.doi else ''}")
    for a in rec.attachments:
        state = "本地可用" if a.exists else ("仅链接" if a.url else "缺失")
        parts.append(f"- 附件[{a.key}] {a.filename}（{a.content_type}，{state}）")
    notes = _collector.item_notes(item_key)
    for n in notes[:5]:
        import re as _re

        text = _re.sub(r"<[^>]+>", " ", n["html"])[:200]
        parts.append(f"- 笔记[{n['key']}] {text}")
    return "\n".join(parts)


@mcp.tool
def zotero_annotations(
    item_key: str, color: str | None = None, kind: str | None = None, limit: int = 50
) -> str:
    """读取某条文献 PDF 上的标注（高亮/批注）。可按颜色（如 #ffd400）与类型过滤。"""
    anns = _collector.item_annotations(item_key)
    if color:
        anns = [a for a in anns if a["color"].lower() == color.lower()]
    if kind:
        anns = [a for a in anns if a["type"] == kind]
    if not anns:
        return "该条目无匹配标注。"
    out = []
    for a in anns[:limit]:
        line = f"[{a['type']}] p.{a['page_label']}"
        if a["color"]:
            line += f" {a['color']}"
        if a["text"]:
            line += f"「{a['text'][:150]}」"
        if a["comment"]:
            line += f" 批注: {a['comment'][:150]}"
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# litflow 核心类
# ---------------------------------------------------------------------------

@mcp.tool
def litflow_prepare(
    collection: str,
    limit: int | None = None,
    use_ocr: bool = True,
    rebuild: bool = False,
    wait: bool = True,
    wait_timeout_sec: int = 3600,
) -> str:
    """将某 Zotero 子分类一键做好 AI 调用准备（采集→转换→语义包→向量索引）。

    默认增量：只处理新增或修改过的条目（依据 Zotero version 与附件指纹）。
    Args:
        collection: Zotero 分类名（可用 zotero_collections 查询）
        limit: 只处理前 N 条（调试用）
        use_ocr: 是否对扫描版 PDF 启用 OCR（内存不足时任务自动推迟）
        rebuild: True 则全量重建（忽略增量）
        wait: 阻塞等待完成；False 立即返回任务 ID 供轮询
        wait_timeout_sec: 等待超时（超时后任务仍在后台继续）
    """
    mgr = get_manager()
    job = mgr.submit("prepare", {
        "collection": collection, "limit": limit,
        "use_ocr": use_ocr, "rebuild": rebuild,
    })
    if not wait:
        return f"任务已提交：{job.id}。用 litflow_job_status 查询进度。"

    deadline = time.time() + wait_timeout_sec
    while time.time() < deadline:
        j = mgr.get(job.id)
        if j and j.status in ("done", "error", "cancelled"):
            return _fmt_job(j)
        time.sleep(2)
    return (
        f"等待超时（任务 {job.id} 仍在后台运行，状态 "
        f"{mgr.get(job.id).status}）。用 litflow_job_status 继续查询。"
    )


@mcp.tool
def litflow_job_status(job_id: str) -> str:
    """查询任务进度与最近日志。"""
    job = get_manager().get(job_id)
    if job is None:
        return f"任务 {job_id} 不存在"
    return _fmt_job(job)


@mcp.tool
def litflow_jobs() -> str:
    """列出全部任务（最近 100 个）。"""
    jobs = get_manager().list()
    if not jobs:
        return "暂无任务"
    lines = []
    for j in jobs:
        lines.append(
            f"{j['id']}  {j['status']:9s}  {j['kind']} {j['params'].get('collection', '')}"
        )
    return "\n".join(lines)


@mcp.tool
def litflow_cancel_job(job_id: str) -> str:
    """取消一个排队/运行中的任务。"""
    ok = get_manager().cancel(job_id)
    return "已取消" if ok else "无法取消（不存在或已结束）"


@mcp.tool
def litflow_search(query: str, collection: str | None = None, top_k: int = 8) -> str:
    """自然语言语义检索已准备的文献（含 adhoc 资料包，collection 形如 adhoc:书库名）。

    返回带原书页码（page 字段）的命中片段，支持跨语言（中文查询命中外文文献）。
    """
    hits = _pipe.search(query, collection=collection, top_k=top_k)
    if not hits:
        return "无结果。请确认该分类已用 litflow_prepare 准备。"
    out = []
    for h in hits:
        page = f" p.{h['page']}" if h.get("page") else ""
        out.append(
            f"《{h['title']}》（{h.get('creator', '')} {h.get('year', '')}）"
            f"[{h['collection']}]{page}\n{h['text'][:500]}"
        )
    return "\n\n---\n\n".join(out)


@mcp.tool
def litflow_find_similar(item_key: str, top_k: int = 5) -> str:
    """找与某条 Zotero 文献语义相似的其他文献块（基于向量索引）。"""
    # 该条目在索引中的所有块 → 取其代表（前 3 块的均值查询）
    t = _pipe.index._get_table()
    if t is None:
        return "向量索引为空。"
    import pyarrow.compute as pc

    tbl = t.to_arrow().filter(pc.equal(tbl_col(tbl, "item_key"), item_key))
    if tbl.num_rows == 0:
        return f"条目 {item_key} 未被索引（其分类可能尚未 prepare）。"
    import numpy as np

    vecs = np.array(tbl.column("vector").to_pylist()[:3], dtype=np.float32)
    qv = (vecs.mean(axis=0) / np.linalg.norm(vecs.mean(axis=0))).tolist()
    q = (
        t.search(qv).metric("cosine").limit(top_k * 6)
        .select(["text", "page", "item_key", "title", "creator", "year", "collection", "source_file"])
    )
    out = []
    for row in q.to_arrow().to_pylist():
        if row["item_key"] == item_key:
            continue
        page = f" p.{row['page']}" if row.get("page") else ""
        out.append(
            f"《{row['title']}》（{row.get('creator', '')} {row.get('year', '')}）"
            f"[{row['collection']}]{page}\n{row['text'][:300]}"
        )
        if len(out) >= top_k:
            break
    return "\n\n---\n\n".join(out) or "未找到相似文献。"


def tbl_col(tbl, name):
    return tbl.column(name)


@mcp.tool
def litflow_manifest(collection: str) -> str:
    """查看某分类的语义文献包清单（条目、转换路由、页数、状态）。"""
    mf = _packager.load_manifest(collection)
    if mf is None:
        return f"分类「{collection}」尚未准备"
    lines = [f"{collection}（更新于 {mf.get('updated_at')}）"]
    for i in mf["items"]:
        notes = f"；{'；'.join(i['notes'])}" if i.get("notes") else ""
        lines.append(
            f"- {i['title']}（{i.get('creator', '')} {i.get('year', '')}）"
            f" [{i['route']}] {i['pages']}页 {i['chars']}字符 {i['status']}{notes}"
        )
    return "\n".join(lines)


@mcp.tool
def litflow_read_item(collection: str, title: str) -> str:
    """按题名（或题名片段/条目目录名）读取某条文献的 markdown 全文。

    全文可能很长；建议先 litflow_search 定位，再读取命中文献。
    """
    mf = _packager.load_manifest(collection)
    if mf is None:
        return f"分类「{collection}」尚未准备"
    target = None
    for i in mf["items"]:
        if title in i["title"] or title in i["dir"]:
            target = i
            break
    if target is None:
        names = "\n".join(f"- {i['title']}" for i in mf["items"])
        return f"未匹配到条目。该分类包含：\n{names}"
    path = os.path.join(target["dir"], "content.md")
    if not os.path.isfile(path):
        return "该条目无内容（转换结果为空）"
    with open(path, encoding="utf-8") as f:
        text = f.read()
    max_chars = 200_000
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n…（已截断，全文 {len(text)} 字符）"
    return text


# ---------------------------------------------------------------------------
# adhoc 临时资料类
# ---------------------------------------------------------------------------

@mcp.tool
def litflow_adhoc_process(
    paths: list[str],
    name: str | None = None,
    use_ocr: bool = True,
    recursive: bool = True,
    wait: bool = True,
    wait_timeout_sec: int = 3600,
) -> str:
    """把任意本地文件/文件夹做成可检索语义包（OCR + 向量索引）。

    适用：非 Zotero 资料，如电子书文件夹、散装 PDF、讲义等。
    支持格式：pdf/epub/docx/html/txt/md/xlsx/pptx 等；扫描版 PDF 自动走 OCR。
    Args:
        paths: 文件或文件夹路径列表
        name: 资料包名称（默认取首个路径的目录名）；检索集合名为 adhoc:<名称>
        use_ocr: 扫描 PDF 是否 OCR
        recursive: 文件夹是否递归处理
        wait: 阻塞等待；False 立即返回任务 ID
    """
    if not paths:
        return "paths 不能为空。"
    for p in paths:
        if not os.path.exists(os.path.expanduser(p)):
            return f"路径不存在: {p}"
    if not name:
        name = os.path.basename(os.path.expanduser(paths[0]).rstrip("/")) or "未命名"
    mgr = get_manager()
    job = mgr.submit("adhoc", {
        "paths": paths, "name": name,
        "use_ocr": use_ocr, "recursive": recursive, "index": True,
    })
    if not wait:
        return f"任务已提交：{job.id}。用 litflow_job_status 查询进度。"
    deadline = time.time() + wait_timeout_sec
    while time.time() < deadline:
        j = mgr.get(job.id)
        if j and j.status in ("done", "error", "cancelled"):
            return _fmt_job(j)
        time.sleep(2)
    return f"等待超时（任务 {job.id} 仍在后台）。用 litflow_job_status 查询。"


@mcp.tool
def litflow_adhoc_list() -> str:
    """列出已建立的 adhoc 资料包及其条目状态。"""
    from .adhoc import AdhocProcessor

    packs = AdhocProcessor(_packager, _pipe.index, _pipe.embedder, _pipe.ocr).list_packs()
    if not packs:
        return "暂无资料包。用 litflow_adhoc_process 创建。"
    return "\n".join(
        f"- {p['name']}：{p['ok_items']}/{p['items']} 条就绪（更新于 {p['updated_at']}）"
        for p in packs
    )


# ---------------------------------------------------------------------------
# 运维类
# ---------------------------------------------------------------------------

@mcp.tool
def litflow_doctor() -> str:
    """诊断系统连通性：Zotero API、LM Studio embedding、输出目录权限。"""
    lines = []
    # Zotero
    try:
        colls = _collector.collections()
        lines.append(f"✓ Zotero 本地 API 在线（{len(colls)} 个分类）")
    except Exception as e:  # noqa: BLE001
        lines.append(f"✗ Zotero 本地 API 不可达：{e}")
    # LM Studio
    try:
        models = _pipe.embedder.embed(["诊断"])
        lines.append(f"✓ LM Studio embedding 在线（{models.shape[1]} 维）")
    except Exception as e:  # noqa: BLE001
        lines.append(f"✗ LM Studio embedding 不可达：{e}")
    # 输出目录
    root = _packager.root
    try:
        os.makedirs(root, exist_ok=True)
        probe = os.path.join(root, ".litflow_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        lines.append(f"✓ 语义包目录可写：{root}")
    except OSError as e:
        lines.append(f"✗ 语义包目录不可写（{root}）：{e}")
    # OCR 后端
    from .ocr_backends import backend_info

    bi = backend_info()
    if bi["backend"] == "http":
        try:
            r = httpx.get(f"{bi['api'].rstrip('/')}/models", timeout=10)
            r.raise_for_status()
            lines.append(f"✓ OCR http 后端在线：{bi['api']}（模型 {bi['model_name']}）")
        except Exception as e:  # noqa: BLE001
            lines.append(f"✗ OCR http 后端不可达：{bi['api']}（{e}）")
    elif os.path.isdir(bi["mlx_model_dir"]):
        lines.append(f"✓ OCR mlx 模型目录存在：{bi['mlx_model_dir']}")
    else:
        lines.append(
            f"✗ OCR mlx 模型目录不存在：{bi['mlx_model_dir']}（扫描件将无法转换；"
            "可改用 http 后端服务 GGUF，见 README「OCR 引擎与模型部署」）"
        )
    return "\n".join(lines)


@mcp.tool
def litflow_system_status() -> str:
    """系统资源状态（内存守卫阈值、CPU、负载）。"""
    s = system_status()
    return (
        f"内存：可用 {s['mem_available_mb'] / 1024:.1f}GB / 共 "
        f"{s['mem_total_mb'] / 1024:.1f}GB（守卫阈值 {s['mem_guard_mb'] / 1024:.0f}GB）\n"
        f"CPU：{s['cpu_percent']}%  负载：{s['loadavg']}"
    )


def _fmt_job(job) -> str:
    lines = [f"任务 {job.id}（{job.kind}）：{job.status}"]
    if job.result:
        r = job.result
        n = r.get("records", r.get("files"))
        lines.append(
            f"结果：共 {n} 条（新增/转换 {r.get('converted')}，跳过 {r.get('skipped', 0)}），"
            f"索引 {r.get('chunks_indexed')} 块，"
            f"OCR {'启用' if r.get('ocr_used') else '未用'}，耗时 {r.get('elapsed_sec')}s"
        )
    if job.error:
        lines.append(f"错误：{job.error}")
    if job.log:
        lines.append("最近日志：")
        lines.extend(f"  {e['msg']}" for e in job.log[-15:])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Web UI 路由注册（与 MCP 同端口）
# ---------------------------------------------------------------------------

for _r in webmod.routes:
    mcp.custom_route(_r.path, methods=_r.methods)(_r.endpoint)


def build_app():
    """构建总 ASGI 应用：Web UI + MCP streamable-http（/mcp）。"""
    return mcp.http_app(path="/mcp")

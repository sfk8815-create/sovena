"""litflow Web 监管界面（Starlette）。

与 MCP 服务同进程运行：`python -m litflow.server`
    - Web UI:      http://localhost:8765/
    - MCP 端点:    http://localhost:8765/mcp   （本地及 tailscale 远程 AI 客户端）
"""
from __future__ import annotations

import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.routing import Route

from .jobs import get_manager, system_status
from .packager import Packager
from .pipeline import Pipeline
from .zotero_collector import ZoteroCollector

UI_PATH = os.path.join(os.path.dirname(__file__), "webui.html")
DEFAULT_PORT = int(os.environ.get("LITFLOW_PORT", "8765"))

_packager = Packager()
_collector = ZoteroCollector()
_search_pipe = Pipeline()


def _adhoc_processor():
    from .adhoc import AdhocProcessor

    return AdhocProcessor(_packager, _search_pipe.index, _search_pipe.embedder, _search_pipe.ocr)


def _safe_name(name: str) -> str:
    """防路径穿越。"""
    return name.replace("/", "_").replace("\\", "_").replace("..", "_")


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------

async def index(_: Request):
    with open(UI_PATH, encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

async def api_collections(_: Request):
    try:
        colls = _collector.collections()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"Zotero 连接失败: {e}"}, status_code=502)
    out = []
    for c in colls:
        name = c["name"]
        mf = _packager.load_manifest(name)
        out.append({
            "key": c["key"],
            "name": name,
            "parent": c.get("parent"),
            "prepared": mf is not None,
            "items": len(mf["items"]) if mf else 0,
            "ok_items": sum(1 for i in mf["items"] if i["status"] == "ok") if mf else 0,
            "updated_at": mf.get("updated_at") if mf else None,
        })
    # adhoc 语义包也纳入（供检索范围/浏览选择）
    for p in _adhoc_processor().list_packs():
        out.append({
            "key": p["id"],
            "name": p["collection"],  # 检索用集合名（adhoc:<名称>）
            "parent": None,
            "prepared": True,
            "items": p["items"],
            "ok_items": p["ok_items"],
            "updated_at": p["updated_at"],
            "adhoc": True,
        })
    return JSONResponse(out)


async def api_manifest(request: Request):
    coll = _safe_name(request.path_params["collection"])
    mf = _packager.load_manifest(coll)
    if mf is None:
        return JSONResponse({"error": "该分类尚未准备"}, status_code=404)
    return JSONResponse(mf)


async def api_item_content(request: Request):
    coll = _safe_name(request.path_params["collection"])
    item = _safe_name(request.path_params["item_dir"])
    path = os.path.join(_packager.collection_dir(coll), item, "content.md")
    if not os.path.isfile(path):
        return PlainTextResponse("未找到内容文件", status_code=404)
    with open(path, encoding="utf-8") as f:
        return PlainTextResponse(f.read())


async def api_item_meta(request: Request):
    coll = _safe_name(request.path_params["collection"])
    item = _safe_name(request.path_params["item_dir"])
    path = os.path.join(_packager.collection_dir(coll), item, "meta.json")
    if not os.path.isfile(path):
        return JSONResponse({"error": "未找到元数据"}, status_code=404)
    import json

    with open(path, encoding="utf-8") as f:
        return JSONResponse(json.load(f))


async def api_jobs_submit(request: Request):
    body = await request.json()
    coll = body.get("collection", "").strip()
    if not coll:
        return JSONResponse({"error": "缺少 collection"}, status_code=400)
    mgr = get_manager()
    job = mgr.submit("prepare", {
        "collection": coll,
        "limit": body.get("limit"),
        "use_ocr": body.get("use_ocr", True),
        "rebuild": body.get("rebuild", False),
    })
    return JSONResponse(job.to_dict(), status_code=201)


# ---------------------------------------------------------------------------
# adhoc（临时资料）
# ---------------------------------------------------------------------------

async def api_adhoc_packs(_: Request):
    return JSONResponse(_adhoc_processor().list_packs())


async def api_adhoc_manifest(request: Request):
    sname = _safe_name(request.path_params["sname"])
    mf = _adhoc_processor().load_manifest(sname)
    if mf is None:
        return JSONResponse({"error": "该资料包不存在"}, status_code=404)
    return JSONResponse(mf)


async def api_adhoc_content(request: Request):
    sname = _safe_name(request.path_params["sname"])
    item_key = _safe_name(request.path_params["item_key"])
    mf = _adhoc_processor().load_manifest(sname)
    if mf is None:
        return PlainTextResponse("资料包不存在", status_code=404)
    for i in mf["items"]:
        if i["item_key"] == item_key:
            path = os.path.join(i["dir"], "content.md")
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    return PlainTextResponse(f.read())
            return PlainTextResponse("该文件无内容", status_code=404)
    return PlainTextResponse("条目不存在", status_code=404)


async def api_adhoc_submit(request: Request):
    body = await request.json()
    paths = [p.strip() for p in body.get("paths", []) if p.strip()]
    name = (body.get("name") or "").strip()
    if not paths:
        return JSONResponse({"error": "缺少 paths（文件/文件夹路径列表）"}, status_code=400)
    if not name:
        name = os.path.basename(paths[0].rstrip("/")) or "未命名"
    for p in paths:
        if not os.path.exists(os.path.expanduser(p)):
            return JSONResponse({"error": f"路径不存在: {p}"}, status_code=400)
    mgr = get_manager()
    job = mgr.submit("adhoc", {
        "paths": paths,
        "name": name,
        "use_ocr": body.get("use_ocr", True),
        "recursive": body.get("recursive", True),
        "index": body.get("index", True),
    })
    return JSONResponse(job.to_dict(), status_code=201)


# ---------------------------------------------------------------------------
# MCP 客户端配置
# ---------------------------------------------------------------------------

async def api_mcp_config(request: Request):
    origin = request.headers.get("host") or f"localhost:{DEFAULT_PORT}"
    cfg = {
        "mcpServers": {
            "litflow": {"url": f"http://{origin}/mcp"}
        }
    }
    return JSONResponse(cfg)


async def api_jobs(_: Request):
    return JSONResponse(get_manager().list())


async def api_job(request: Request):
    job = get_manager().get(request.path_params["job_id"])
    if job is None:
        return JSONResponse({"error": "job 不存在"}, status_code=404)
    return JSONResponse(job.to_dict())


async def api_job_cancel(request: Request):
    ok = get_manager().cancel(request.path_params["job_id"])
    return JSONResponse({"cancelled": ok})


def api_search(request: Request):  # 同步 endpoint：starlette 自动放线程池
    q = request.query_params.get("q", "").strip()
    if not q:
        return JSONResponse({"error": "缺少 q"}, status_code=400)
    coll = request.query_params.get("collection") or None
    top_k = int(request.query_params.get("top_k", "8"))
    try:
        hits = _search_pipe.search(q, collection=coll, top_k=top_k)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"error": f"检索失败（embedding 服务是否在线/密钥正确？）: {e}"},
            status_code=502,
        )
    return JSONResponse({"query": q, "collection": coll, "hits": hits})


async def api_system(_: Request):
    return JSONResponse(system_status())


# ---------------------------------------------------------------------------
# 配置与文件系统浏览（供 Web 路径选择器）
# ---------------------------------------------------------------------------

async def api_config(_: Request):
    """只读配置信息（修改需设置环境变量后重启服务）。"""
    from .indexer import DEFAULT_DB_PATH, DEFAULT_EMBED_API, DEFAULT_EMBED_MODEL
    from .ocr_backends import backend_info
    from .zotero_collector import DEFAULT_API

    return JSONResponse({
        "root": _packager.root,
        "lancedb": DEFAULT_DB_PATH,
        "zotero_api": DEFAULT_API,
        "embed_api": DEFAULT_EMBED_API,
        "embed_model": DEFAULT_EMBED_MODEL,
        "ocr": backend_info(),
        "host": os.environ.get("LITFLOW_HOST", "0.0.0.0"),
        "port": DEFAULT_PORT,
    })


async def api_fs_list(request: Request):
    """列目录（路径选择器用）：?path=/ 默认根目录。跳过隐藏项。"""
    p = request.query_params.get("path") or "/"
    p = os.path.abspath(os.path.expanduser(p))
    if p != "/" and not os.path.isdir(p):
        return JSONResponse({"error": f"不是目录: {p}"}, status_code=400)
    parent = os.path.dirname(p) or "/"
    dirs: list[str] = []
    files: list[str] = []
    try:
        with os.scandir(p) as it:
            for e in it:
                if e.name.startswith("."):
                    continue
                try:
                    if e.is_dir(follow_symlinks=True):
                        dirs.append(e.name)
                    elif e.is_file():
                        files.append(e.name)
                except OSError:
                    continue
    except PermissionError:
        return JSONResponse({"error": f"无权限访问: {p}"}, status_code=403)
    except OSError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({
        "path": p,
        "parent": parent if p != "/" else None,
        "dirs": sorted(dirs),
        "files": sorted(files),
    })


routes = [
    Route("/", index),
    Route("/api/collections", api_collections),
    Route("/api/config", api_config),
    Route("/api/fs/list", api_fs_list),
    Route("/api/manifest/{collection}", api_manifest),
    Route("/api/item/{collection}/{item_dir}/content", api_item_content),
    Route("/api/item/{collection}/{item_dir}/meta", api_item_meta),
    Route("/api/jobs", api_jobs, methods=["GET"]),
    Route("/api/jobs", api_jobs_submit, methods=["POST"]),
    Route("/api/jobs/{job_id}", api_job),
    Route("/api/jobs/{job_id}/cancel", api_job_cancel, methods=["POST"]),
    Route("/api/search", api_search),
    Route("/api/system", api_system),
    Route("/api/adhoc/packs", api_adhoc_packs),
    Route("/api/adhoc/manifest/{sname}", api_adhoc_manifest),
    Route("/api/adhoc/content/{sname}/{item_key}", api_adhoc_content),
    Route("/api/adhoc/submit", api_adhoc_submit, methods=["POST"]),
    Route("/api/mcp-config", api_mcp_config),
]


def create_web_app() -> Starlette:
    return Starlette(routes=routes)

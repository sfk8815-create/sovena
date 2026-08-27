"""sovena 服务总入口：Web 监管界面 + MCP 端点，单进程按需启动。

    python -m sovena.server            # 前台运行，Ctrl+C 停止
    SOVENA_HOST=0.0.0.0 SOVENA_PORT=8765

启动后：
    Web UI:   http://localhost:8765/
    MCP:      http://localhost:8765/mcp   （本地 AI 客户端；
              远程经 tailscale 用 http://<机器IP>:8765/mcp）

本地个人配置可写进项目根目录的 `.env`（KEY=VALUE，已被 .gitignore 忽略，
不会进入仓库），在导入其余模块前加载，例如：
    SOVENA_ROOT=/Volumes/your-disk/zotero_AI
    SOVENA_ZOTERO_API=http://localhost:23119/api

「听需调用」：本服务不常驻、不开机自启；需要监管或 AI 客户端要调用时启动，
用完关闭即可。检索/准备等重操作均在内部串行调度并带内存守卫。
"""
from __future__ import annotations

import os


def _load_dotenv() -> None:
    """加载项目根目录 .env（存在才读；不覆盖已设置的环境变量）。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except OSError:
        pass


_load_dotenv()

import uvicorn  # noqa: E402

from .mcp_server import DEFAULT_HOST, DEFAULT_PORT, build_app  # noqa: E402


def main():
    app = build_app()
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT, log_level="info")


if __name__ == "__main__":
    main()

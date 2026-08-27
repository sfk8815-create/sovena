"""litflow 服务总入口：Web 监管界面 + MCP 端点，单进程按需启动。

    python -m litflow.server            # 前台运行，Ctrl+C 停止
    LITFLOW_HOST=0.0.0.0 LITFLOW_PORT=8765

启动后：
    Web UI:   http://localhost:8765/
    MCP:      http://localhost:8765/mcp   （本地 AI 客户端；
              远程经 tailscale 用 http://<机器IP>:8765/mcp）

「听需调用」：本服务不常驻、不开机自启；需要监管或 AI 客户端要调用时启动，
用完关闭即可。检索/准备等重操作均在内部串行调度并带内存守卫。
"""
from __future__ import annotations

import uvicorn

from .mcp_server import DEFAULT_HOST, DEFAULT_PORT, build_app


def main():
    app = build_app()
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT, log_level="info")


if __name__ == "__main__":
    main()

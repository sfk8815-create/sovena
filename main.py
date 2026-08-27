"""sovena 一键启动入口。

    uv run sovena          # 或: python main.py

等价于 `python -m sovena.server`：单进程启动 Web 监管界面 + MCP 端点。
环境变量见 sovena/server.py 模块说明（SOVENA_HOST / SOVENA_PORT /
SOVENA_ROOT / SOVENA_ZOTERO_API 等）。
"""
from sovena.server import main

if __name__ == "__main__":
    main()

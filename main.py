"""litflow 一键启动入口。

    uv run litflow          # 或: python main.py

等价于 `python -m litflow.server`：单进程启动 Web 监管界面 + MCP 端点。
环境变量见 litflow/server.py 模块说明（LITFLOW_HOST / LITFLOW_PORT /
LITFLOW_ROOT / LITFLOW_ZOTERO_API 等）。
"""
from litflow.server import main

if __name__ == "__main__":
    main()

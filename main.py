"""jinyun 一键启动入口。

    uv run jinyun          # 或: python main.py

等价于 `python -m jinyun.server`：单进程启动 Web 监管界面 + MCP 端点。
环境变量见 jinyun/server.py 模块说明（JINYUN_HOST / JINYUN_PORT /
JINYUN_ROOT / JINYUN_ZOTERO_API 等）。
"""
from jinyun.server import main

if __name__ == "__main__":
    main()

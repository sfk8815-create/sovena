# litflow

**Zotero → Markdown 语义文献包 → 向量检索**：面向人文研究的本地文献流处理系统，并以 MCP（Model Context Protocol）服务的形式暴露给任意本地/远程 AI 客户端。

对标 [cookjohn/zotero-mcp](https://github.com/cookjohn/zotero-mcp)，并在其基础上扩展为完整的「文献流」：

| 能力 | zotero-mcp | litflow |
| --- | --- | --- |
| Zotero 分类/条目/标注读取 | ✅ | ✅（本地 API，无需 Web API key） |
| 文献全文获取 | 链接附件路径 | ✅ 转换为 AI 友好 Markdown（含【书页页码】标注） |
| 扫描/影印 PDF OCR | ❌ | ✅ L2 通道（Unlimited-OCR-MLX，Apple Silicon） |
| 向量语义检索 | ❌ | ✅ LanceDB + LM Studio embedding |
| 增量处理 | ❌ | ✅ 条目 version + 附件指纹（mtime）双重检测 |
| 一键启动 | 多步配置 | ✅ `uv run litflow` 单进程起 Web + MCP |
| 非 Zotero 资料（任意文件夹） | ❌ | ✅ adhoc 临时资料包（OCR + 索引） |
| 远程部署 | ❌ | ✅ 服务端起一次，其他电脑填 URL 即可（Tailscale） |
| Zotero 客户端集成 | ❌ | ✅ 附带 Zotero 插件（.xpi，分类/条目右键直达） |
| 任务调度/资源守卫 | ❌ | ✅ OCR 并发=1、内存守卫、按需加载模型 |

## 架构

```
Zotero(本地API) ─┐
                 ├─ Pipeline.prepare ─┬─ L1 文本路(pymupdf) ─┐
任意文件/文件夹 ─┘  (增量)             ├─ L2 OCR路(MLX)      ├─ 语义包(content.md+meta.json)
                                     └─ anydoc(非PDF)      ┘        │
                                                                 LanceDB 向量索引
                                                                     │
                              ┌──────────────────────────────────────┤
                              │                                      │
                        Web 监管台(:8765)                      MCP 端点(/mcp)
                       (HTMX/原生JS)                    (本地 & Tailscale 远程 AI 客户端)
```

- **L1 文本路**：有文本层的 PDF → pymupdf 提取，页码标注优先用 PDF Page Labels（书页页码，非物理页序）
- **L2 OCR 路**：扫描/影印件 → MLX OCR 结构化识别，页码优先级：OCR 识别的 page_number > PDF Page Label > 物理页序
- **非 PDF**：docx / epub / html / txt / md / xlsx / pptx 等 → anydoc / trafilatura
- **检索**：LM Studio 提供 embedding（如 qwen3-embedding），LanceDB 本地向量库

## 快速开始

要求：macOS（Apple Silicon，OCR 通道依赖 MLX）、Python ≥ 3.12、[uv](https://docs.astral.sh/uv/)。

### 1. 前置条件

- **Zotero** 桌面版运行中，并启用本地 API（默认 `http://localhost:23119/api`，只读）
- **LM Studio** 加载 embedding 模型（默认 `qwen3-embedding-4b`）并开启本地服务

### 2. 安装与一键启动

```bash
git clone https://github.com/<you>/litflow.git
cd litflow
uv sync

uv run litflow            # 一键启动（等价 python main.py / python -m litflow.server）
```

启动后：

- Web 监管台：http://localhost:8765/
- MCP 端点：`http://localhost:8765/mcp`

> 服务**按需运行**：不常驻、不开机自启，用完 Ctrl+C 关闭。重操作在内部串行调度（OCR 并发=1，带内存守卫），防止机器过载。

### 3. AI 客户端接入（其他电脑同样适用）

任意支持 MCP streamable-http 的客户端（Claude Desktop、Cherry Studio、Trae 等）填入：

```json
{
  "mcpServers": {
    "litflow": { "url": "http://localhost:8765/mcp" }
  }
}
```

远程（其他电脑）部署：服务端 `LITFLOW_HOST=0.0.0.0` 启动（默认），客户端把 URL 换成 `http://<服务器IP或Tailscale主机名>:8765/mcp` 即可。Web 监管台「设置」页可一键复制/下载当前部署的配置 JSON。

## Zotero 插件（可选）

`dist/litflow-plugin-<version>.xpi` 是可安装到 **Zotero 7+（含 9/10）** 的客户端插件，让 Zotero 内直达 litflow：

- **分类右键** → 「litflow：准备/更新语义包（增量）」
- **条目右键** → 「litflow：把附件加入临时语义包」（所选条目的本地文件附件走 adhoc 流程）
- **工具菜单 litflow** → 打开监管台 / 服务端地址设置 / 复制 MCP 客户端配置 / 连接检查

安装：Zotero → 工具 → 插件 → 右上角齿轮 → Install Plugin From File… → 选择 `dist/litflow-plugin-0.1.0.xpi`。默认连 `http://localhost:8765`；**其他电脑**在「工具 → litflow → 服务端地址…」填 litflow 服务器地址即可。

重新打包插件（修改 `zotero-plugin/` 后）：

```bash
bash zotero-plugin/build.sh    # 产出 dist/litflow-plugin-<version>.xpi
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LITFLOW_ROOT` | `/Volumes/macstudio-work/synology_drive/zotero_AI` | 语义文献包根目录（其他机器部署时必须改；LanceDB 默认随之） |
| `LITFLOW_ZOTERO_API` | `http://localhost:23119/api` | Zotero 本地 API |
| `LITFLOW_HOST` | `0.0.0.0` | 服务监听地址 |
| `LITFLOW_PORT` | `8765` | 服务端口 |
| `LITFLOW_LMSTUDIO` | LM Studio 本地 embedding 地址 | 向量化服务 |
| `LITFLOW_EMBED_MODEL` | `text-embedding-qwen3-embedding-4b` | embedding 模型名 |
| `LITFLOW_LANCEDB` | `$LITFLOW_ROOT/_lancedb` | 向量库目录 |
| `LITFLOW_MEM_GUARD_MB` | `12288` | 内存守卫阈值（MB），低于则推迟新任务 |

## 使用

### Zotero 文献流（增量）

Web 台选择分类 →「启动准备」；或让 AI 客户端调用 MCP 工具 `litflow_prepare`。重复执行自动增量：仅处理新增/变更条目（Zotero version 变化或附件 mtime 变化），索引按条目级增删。勾选「全量重建」可强制重来。

### adhoc 临时资料（任意文件/文件夹）

把电子书库、散装 PDF、讲义等做成可检索语义包：

- **Web 台**：「临时资料包」卡片填路径（多个用换行或 `;` 分隔）→ 提交，之后可与 Zotero 分类一起被语义检索
- **MCP**：`litflow_adhoc_process(paths=["/Volumes/.../E_book/某子目录"], name="我的书库")`
- **REST**：`POST /api/adhoc/submit` `{"paths": [...], "name": "..."}`

支持 pdf/epub/docx/html/txt/md/xlsx/pptx 等；扫描版 PDF 自动走 OCR；同样支持增量（源文件 mtime 不变则跳过）。

### MCP 工具一览

| 分类 | 工具 |
| --- | --- |
| Zotero 读取 | `zotero_collections` `zotero_search` `zotero_item` `zotero_annotations` |
| 文献流 | `litflow_prepare` `litflow_manifest` `litflow_read_item` `litflow_search` `litflow_find_similar` |
| adhoc | `litflow_adhoc_process` `litflow_adhoc_list` |
| 任务/运维 | `litflow_job_status` `litflow_jobs` `litflow_cancel_job` `litflow_system_status` `litflow_doctor` |

> 注：Zotero 本地 API 为只读，故不提供写操作工具。

## 语义包目录结构

```
$LITFLOW_ROOT/
  <分类名>/
    _manifest.json                 # 分类级清单（增量依据）
    <作者>_<年份>_<标题>/
      meta.json                    # Zotero 元数据 + 转换统计
      content.md                   # AI 友好 markdown（含【书页页码】标注）
  adhoc/
    <资料包名>/
      _manifest.json
      <文件名slug>/
        meta.json
        content.md
  _lancedb/                        # 向量库
```

## REST API 概览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/collections` | Zotero 分类 + adhoc 包及准备状态 |
| GET | `/api/config`、`/api/system` | 运行配置、系统状态（内存/CPU/磁盘/任务） |
| GET | `/api/fs/list?path=` | 服务端目录列表（Web 路径选择器用） |
| POST | `/api/jobs` | 提交 prepare 任务（collection/limit/use_ocr/rebuild） |
| POST | `/api/adhoc/submit` | 提交 adhoc 任务（paths/name/use_ocr/recursive） |
| GET | `/api/jobs[/{id}]` | 任务列表/详情（含日志），POST `/{id}/cancel` 取消 |
| GET | `/api/search?q=` | 语义检索（可限定 collection） |
| GET | `/api/manifest/{collection}`、`/api/adhoc/manifest/{name}` | 清单 |
| GET | `/api/item/{collection}/{dir}/content` 等 | 内容/元数据 |
| GET | `/api/system`、`/api/mcp-config` | 系统状态、MCP 客户端配置 |

## 项目结构

```
litflow/
  main.py                  # 一键启动入口
  litflow/
    server.py              # 服务总入口（Web + MCP 同进程）
    web.py / webui.html    # Web 监管台（分区 Tab + 路径选择器）
    mcp_server.py          # MCP 工具集
    zotero_collector.py    # Zotero 本地 API 采集（附件 4 路解析）
    pipeline.py            # prepare 流水线（增量）
    adhoc.py               # 任意资料临时处理
    converter.py           # L1/L2/anydoc 转换
    indexer.py             # 分块 + 向量化 + LanceDB
    packager.py            # 语义包落盘
    jobs.py                # 任务调度（内存守卫/OCR 并发=1）
  zotero-plugin/           # Zotero 客户端插件源码（bootstrap 结构）
  dist/                    # 构建产物（litflow-plugin-<version>.xpi）
  ocr_port/                # Unlimited-OCR-MLX（MLX OCR 引擎）
  tests/                   # 开发期验证脚本（P0 环境探针/样张测试）
```

## License

MIT

# litflow

**Zotero → Markdown 语义文献包 → 向量检索**：面向各学科学术研究的本地文献流处理系统（人文、社科、理工、医学等均有大量影印/扫描资料的领域尤其受益），并以 MCP（Model Context Protocol）服务的形式暴露给任意本地/远程 AI 客户端。

## 应用场景

- **文献综述与写作**：把 Zotero 里的文献（含影印古籍、扫描版外文 PDF）批量转成带书页页码的 Markdown，AI 客户端引用时可精确溯源到页
- **跨库语义检索**：对几百篇文献用自然语言提问（如「古琴音色的声学测量方法有哪些」），而不是逐篇翻找关键词
- **古籍/影印本数字化**：扫描版 PDF 自动走 OCR 通道，还原标题、表格、双栏版式为结构化文本
- **Zotero 之外的资料**：电子书库、散装 PDF、讲义等任意文件夹，做成同样可检索的临时资料包（adhoc）
- **AI 深度阅读外脑**：Claude Desktop / Cherry Studio / Trae 等客户端经 MCP 直接查你的文献库，回答带出处
- **远程协作**：服务端部署在任意满足运行条件的电脑上（家中/实验室/云端均可），其他电脑在 AI 客户端里填服务端 URL 即可使用

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

## OCR 引擎与模型部署

litflow 的 OCR 通道使用 **Unlimited-OCR**（[百度开源](https://github.com/baidu/Unlimited-OCR)，MIT 协议，模型权重在 [HuggingFace `baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR)）：DeepSeek-V2 MoE 解码器 + SAM/CLIP 双视觉塔的文档 OCR 模型，能整篇识别多页扫描件并还原标题/表格/版式。本仓库的 `ocr_port/` 是其 **MLX 移植**（来自 [mlx-vlm 社区实现](https://github.com/Blaizzy/mlx-vlm)），在 Apple Silicon 上原生跑、无需显卡/CUDA。

**部署方式（二选一）**：

1. **LM Studio 下载（推荐，最简单）**：打开 LM Studio → 搜索 `Unlimited-OCR-MLX`（作者 LoJexLLM 上传的 MLX 权重格式）→ 下载。模型会落在 `~/.lmstudio/models/LoJexLLM/Unlimited-OCR-MLX/`，正是 litflow 的默认路径，**无需任何配置**。
2. **HuggingFace 手动下载**：从 HuggingFace 下载 MLX 权重目录，放到任意位置，然后在 `.env` 里指定：

```dotenv
LITFLOW_OCR_MODEL=/path/to/Unlimited-OCR-MLX
```

> OCR 模型约 7GB（fp16），只在遇到扫描件时才加载进内存，转换完立即释放（引擎按需加载，平时不占内存）。若不处理扫描件，可不装此模型（Web 台取消勾选「启用 OCR」即可）。

**Embedding 模型（检索用，必须）**：LM Studio → 搜索 `qwen3-embedding-4b` → 下载并加载 → 保持 LM Studio 本地服务开启（默认 `http://localhost:1234/v1`）。litflow 经 HTTP 调用它做向量化。

## 快速开始（新手保姆级）

要求：任意电脑均可部署，核心流程只需 Python ≥ 3.12（Windows / macOS / Linux 通用）；**OCR 通道**依赖 MLX 框架，仅支持 Apple Silicon Mac。全程只需复制粘贴命令。

### 第 1 步：装 uv（Python 包管理器，一次性）

打开「终端」（启动台搜索「终端」或 Terminal），粘贴：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

装完后**关闭终端，重新打开**（让命令生效）。验证：`uv --version` 能出版本号即可。

### 第 2 步：装 Zotero 并保持运行

- 到 [zotero.org](https://www.zotero.org/) 下载安装 Zotero 7+，导入你的文献
- litflow 通过 Zotero 的**本地 API** 读取（Zotero 打开着就自动可用，无需任何设置）
- 附件可以是「导入的附件」或「链接附件」，两种都支持

### 第 3 步：装 LM Studio 并下载两个模型（检索必需 + OCR 可选）

到 [lmstudio.ai](https://lmstudio.ai/) 下载安装 LM Studio：

1. **embedding 模型（检索必需）**：LM Studio 里搜索 `qwen3-embedding-4b` → 下载 → 加载到内存 → 顶部「Developer」标签确认本地服务已启动
2. **OCR 模型（仅扫描件需要）**：搜索 `Unlimited-OCR-MLX` → 下载即可（litflow 需要时会自己加载/释放，不用在 LM Studio 里常驻）

> 只想快速体验、暂时不检索？LM Studio 可以后补，先跳过做第 4-5 步。

### 第 4 步：获取 litflow 并安装依赖

```bash
git clone https://github.com/<you>/litflow.git
cd litflow
uv sync        # 自动下载全部依赖（首次约 1.3GB，需要几分钟）
```

> 非 Apple Silicon 电脑（Windows / Linux / Intel Mac）：MLX 相关依赖仅在用到 OCR 通道时才需要，`uv sync` 在这些平台会自动跳过或用 CPU 兼容版本安装；文本 PDF、非 PDF 文档转换、检索等核心功能均可正常使用。

### 第 5 步：一键启动

```bash
uv run litflow
```

看到 `Uvicorn running on http://0.0.0.0:8765` 即成功。浏览器打开 **http://localhost:8765**：

- 「性能监控」页的圆点是绿色 → 服务正常
- 「Zotero 文献流」下拉框能看到你的分类 → Zotero 连通

用完在终端按 **Control + C** 停止。服务**不常驻、不开机自启**，重操作内部串行调度（OCR 并发=1、内存守卫），不会把电脑跑死。

### 第 6 步（可选）：个人路径配置

默认数据存在 `~/litflow_data`。想放别处（如移动硬盘），在项目根目录建 `.env` 文件：

```bash
echo 'LITFLOW_ROOT=/Volumes/你的盘/litflow_data' > .env
```

`.env` 已被 git 忽略，写个人路径不会进仓库。

### 第 7 步：跑第一个任务

Web 台「Zotero 文献流」→ 选一个**小分类**（如 5 条文献）→「启动准备」→ 切到「性能监控」看进度。完成后去「语义检索」问个问题试试。

### 常见问题

| 现象 | 解决 |
| --- | --- |
| 提示 `uv: command not found` | 第 1 步的 uv 没装好或没重开终端 |
| 提示端口被占用 | 旧服务没关：`lsof -ti tcp:8765 \| xargs kill` 后重启 |
| 「Zotero 连接失败」 | Zotero 没打开，或装的是旧版（需 7.0+） |
| 检索报错/无结果 | LM Studio 没开服务、embedding 模型没加载；或该分类还没「准备」过 |
| OCR 报模型错误 | 没下载 `Unlimited-OCR-MLX`，或路径不对（见上节） |
| 机器风扇狂转 | 正常：OCR 任务较重；任务结束模型会自动卸载 |

### 第 8 步（可选）：AI 客户端接入（其他电脑同样适用）

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

所有配置均可用环境变量设置；**推荐**在项目根目录建一个 `.env` 文件（已被 `.gitignore` 忽略，适合放个人路径），服务启动时自动加载：

```dotenv
LITFLOW_ROOT=/Volumes/your-disk/zotero_AI
LITFLOW_ZOTERO_API=http://localhost:23119/api
```

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LITFLOW_ROOT` | `~/litflow_data` | 语义文献包根目录（**个人部署建议在 `.env` 里设置**；LanceDB 默认随之） |
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
- **MCP**：`litflow_adhoc_process(paths=["/path/to/E_book/某子目录"], name="我的书库")`
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
    server.py              # 服务总入口（Web + MCP 同进程，自动加载 .env）
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
  .env                     # 本地个人配置（可选，不入库）
```

## 致谢

litflow 站在以下项目肩膀上，深表感谢：

- **[Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)**（百度，MIT）— 文档 OCR 模型本体；`ocr_port/` 代码移植自 [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 社区的 MLX 实现，权重经 LM Studio（作者 LoJexLLM 整理的 MLX 格式）分发
- **[cookjohn/zotero-mcp](https://github.com/cookjohn/zotero-mcp)** — 本项目对标与超越的起点
- **[Zotero](https://www.zotero.org/)**（AGPL）— 文献管理本体与本地 API
- **[PyMuPDF](https://github.com/pymupdf/PyMuPDF)**（AGPL）— PDF 文本提取与页码标签
- **[LanceDB](https://github.com/lancedb/lancedb)**（Apache-2.0）— 本地向量库
- **[FastMCP](https://github.com/jlowin/fastmcp)**（MIT）— MCP 服务框架
- **[anydoc](https://pypi.org/project/firecrawl-anydoc/)**（firecrawl-anydoc）— docx/epub 等非 PDF 文档转换
- **[trafilatura](https://github.com/adbar/trafilatura)**（Apache-2.0）— 网页正文提取
- **[LM Studio](https://lmstudio.ai/)** — 本地模型运行时（embedding 向量化服务）
- **[uv](https://docs.astral.sh/uv/)**（Astral，MIT）— Python 包管理

## License

MIT

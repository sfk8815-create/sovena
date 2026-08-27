# Sovena

**Zotero → Markdown semantic literature packs → vector retrieval**: a local literature-flow processing system for academic research across all disciplines (especially beneficial for fields rich in scanned/photocopied materials — humanities, social sciences, STEM, medicine), exposed to any local or remote AI client as an MCP (Model Context Protocol) service.

> "Sovena" echoes 缙云文采 in Chinese: 缙云 (Jinyun) is the mountain in Beibei, Chongqing, home of Southwest University; 文采 carries a double meaning — to **harvest** the essence of literature, and the **brilliance** of fine writing.
> "Pluck the flowers of the distant past, to brew the honey of our own." — Wu Mi (who taught at Southwest University for twenty-eight years)

[中文版 README](README.md)

## Use Cases

- **Literature review & writing**: batch-convert Zotero literature (including scanned ancient books and photocopied PDFs) into Markdown with printed-page numbers, so AI clients can cite with precise page-level provenance
- **Cross-library semantic search**: ask natural-language questions (e.g. "what are the acoustic measurement methods for timbre?") across hundreds of papers, instead of hunting keywords one by one
- **Digitizing ancient books / photocopies**: scanned PDFs automatically go through the OCR pipeline, restoring headings, tables, and two-column layouts as structured text
- **Materials beyond Zotero**: e-book libraries, loose PDFs, lecture notes — any folder becomes an equally searchable ad-hoc pack
- **A deep-reading external brain for AI**: Claude Desktop / Cherry Studio / Trae and other MCP clients query your library directly, with cited sources
- **Remote collaboration**: the server runs on any machine that meets the requirements (home / lab / cloud); other computers simply fill in the server URL in their AI client

## Core Capabilities

| Capability | Description |
| --- | --- |
| Deep Zotero integration | Read collections/items/annotations via the local API (no Web API key needed); ships with a Zotero plugin (.xpi, right-click from any collection/item) |
| Full-text conversion | Batch-convert attachments into AI-friendly Markdown with printed-page-number markers (PDF Page Labels + OCR page numbers as dual sources) |
| Scanned-document OCR | Unlimited-OCR structured recognition; MLX / GGUF (llama-server) dual backends, all platforms, can be remote |
| Vector semantic search | LanceDB + any OpenAI-compatible embedding service (local or remote commercial platforms) |
| Incremental processing | Dual detection via item version + attachment fingerprint (mtime); only new/changed content is processed |
| One-command startup | `uv run sovena` starts the web console + MCP endpoint in a single process |
| Non-Zotero materials | E-book libraries, loose PDFs, lecture notes — any files/folders become equally searchable ad-hoc packs |
| Remote deployment | Start the server once; other machines just fill in the URL (e.g. over a Tailscale network) |
| Job scheduling / resource guard | OCR concurrency = 1, memory guard, models loaded/released on demand — your machine never gets buried |

## Architecture

```
Zotero (local API) ─┐
                    ├─ Pipeline.prepare ─┬─ L1 text path (pymupdf) ─┐
Any files/folders ──┘  (incremental)     ├─ L2 OCR path (MLX/GGUF) ├─ semantic packs (content.md + meta.json)
                                         └─ anydoc (non-PDF)       ┘        │
                                                                     LanceDB vector index
                                                       (embedding: local/remote OpenAI-compatible service)
                                                                                │
                              ┌────────────────────────────────────────────────┤
                              │                                                │
                       Web console (:8765)                                MCP endpoint (/mcp)
                      (HTMX / vanilla JS)                          (local & Tailscale remote AI clients)
```

- **L1 text path**: PDFs with a text layer → pymupdf extraction; page markers prefer PDF Page Labels (printed page numbers, not physical page order)
- **L2 OCR path**: scanned/photocopied documents → Unlimited-OCR structured recognition (MLX / GGUF backends); page-number priority: OCR-recognized page_number > PDF Page Label > physical page order
- **Non-PDF**: docx / epub / html / txt / md / xlsx / pptx etc. → anydoc / trafilatura
- **Retrieval**: any OpenAI-compatible embedding service (local mlx-lm / Ollama, or remote platforms like Alibaba Bailian / OpenRouter) + a local LanceDB vector store

## OCR Engine & Model Deployment

Sovena's OCR pipeline uses **Unlimited-OCR** ([open-sourced by Baidu](https://github.com/baidu/Unlimited-OCR), MIT license, weights on [HuggingFace `baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR)): a document-OCR model combining a DeepSeek-V2 MoE decoder with SAM/CLIP dual vision towers, capable of recognizing entire multi-page scans and restoring headings/tables/layout.

Two **backends** are supported; both output the same structured format, fully transparent to the conversion pipeline:

| Backend | How it runs | Platforms |
| --- | --- | --- |
| `mlx` (default) | This repo's `ocr_port/` ([MLX implementation from the mlx-vlm community](https://github.com/Blaizzy/mlx-vlm)) | Apple Silicon Macs |
| `http` | OpenAI-compatible API: llama-server / vLLM serving the **GGUF quantized** model | **Any platform** (Windows / Linux / Intel Macs; pure CPU is fine) |

### Backend 1: MLX (Apple Silicon, default)

Download the MLX weights from HuggingFace ([LoJexLLM/Unlimited-OCR-MLX](https://huggingface.co/LoJexLLM/Unlimited-OCR-MLX)):

```bash
huggingface-cli download LoJexLLM/Unlimited-OCR-MLX \
  --local-dir ~/models/Unlimited-OCR-MLX
```

This lands exactly in Sovena's default path (`~/models/Unlimited-OCR-MLX`) — **zero configuration needed**. Put it elsewhere and specify it in `.env`:

```dotenv
SOVENA_OCR_MODEL=/path/to/Unlimited-OCR-MLX
```

### Backend 2: GGUF (any computer, incl. GPU-less Windows/Linux)

Unlimited-OCR has a community GGUF quantization ([HuggingFace `sahilchachra/Unlimited-OCR-GGUF`](https://huggingface.co/sahilchachra/Unlimited-OCR-GGUF); download the main model, e.g. `Unlimited-OCR-Q4_K_M.gguf` (~3.2GB), plus the vision projector `mmproj-Unlimited-OCR-F16.gguf`), then serve it locally with llama.cpp's llama-server:

```bash
# 1. Download the model
huggingface-cli download sahilchachra/Unlimited-OCR-GGUF \
  Unlimited-OCR-Q4_K_M.gguf mmproj-Unlimited-OCR-F16.gguf --local-dir ./ocr-models

# 2. Start an OpenAI-compatible service on port 8080 (any platform; add GPU flags if available)
llama-server -m ocr-models/Unlimited-OCR-Q4_K_M.gguf \
  --mmproj ocr-models/mmproj-Unlimited-OCR-F16.gguf \
  --host 127.0.0.1 --port 8080

# 3. Enable the http backend on Sovena's side (project root .env)
echo 'SOVENA_OCR_API=http://127.0.0.1:8080/v1' >> .env
```

Any OpenAI-compatible server that can run the GGUF works too (vLLM etc.; set `SOVENA_OCR_MODEL_NAME=...` if the model name differs, and `SOVENA_OCR_API_KEY=...` if authentication is required). The OCR service can even live on a separate GPU machine — just point Sovena at its address.

> Tip: the Q4 quantization is ~3GB and runs fine on an ordinary 16GB machine. Sovena calls it on demand (page by page), so it consumes no memory inside the Sovena process.

**Embedding service (for retrieval, required)** — any OpenAI-compatible `/embeddings` API; pick one:

- **Local** (recommended, free & private): [mlx-lm](https://github.com/ml-explore/mlx-lm) (the official Apple MLX-ecosystem inference server, MIT):

```bash
uv tool install mlx-lm            # or: pip install mlx-lm
huggingface-cli download Qwen/Qwen3-Embedding-4B --local-dir ~/models/Qwen3-Embedding-4B
mlx_lm.server --model ~/models/Qwen3-Embedding-4B --port 8080
# Starts an OpenAI-compatible /v1/embeddings service — Sovena's default address http://localhost:8080/v1
```

Ollama / vLLM and other local OpenAI-compatible options work the same way (set `SOVENA_EMBED_API` if the address differs).
- **Remote commercial platforms** (skip running models locally): Alibaba Bailian / OpenRouter / SiliconFlow etc. — just fill in the API address and key in `.env`:

```dotenv
# Example: Alibaba Cloud Bailian (OpenAI-compatible endpoint)
SOVENA_EMBED_API=https://dashscope.aliyuncs.com/compatible-mode/v1
SOVENA_EMBED_API_KEY=sk-your-key
SOVENA_EMBED_MODEL=text-embedding-v4
```

> Note: switching embedding services/models changes the vector dimension and semantic space, so existing indexes must be rebuilt (Sovena detects dimension mismatches and tells you explicitly; delete the `_lancedb` directory and re-prepare each collection, or tick "full rebuild").

## Quick Start

Requirements: any computer can deploy; the core flow only needs Python ≥ 3.12 (Windows / macOS / Linux). **OCR, pick one**: on Apple Silicon use the default MLX backend (zero config); on other platforms (or to run OCR on a GPU server) use the GGUF backend (see "Backend 2" above). Everything is copy-paste.

### Step 1: Install uv (the Python package manager, one-time)

Open "Terminal" (search for Terminal in Launchpad) and paste:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, **close the terminal and reopen it** (so the command takes effect). Verify with `uv --version`.

### Step 2: Install Zotero and keep it running

- Download and install Zotero 7+ from [zotero.org](https://www.zotero.org/), then import your literature
- Sovena reads via Zotero's **local API** (works automatically while Zotero is open — no setup needed)
- Attachments can be "imported" or "linked" — both are supported

### Step 3: Prepare model services (embedding required + OCR optional)

**Embedding service (required for retrieval)**, pick one:
- **Local (recommended)**: [mlx-lm](https://github.com/ml-explore/mlx-lm) (official Apple MLX ecosystem, MIT) — `uv tool install mlx-lm`, download the model, and run `mlx_lm.server --model <model-dir> --port 8080` (full commands in the "Embedding service" section above). Ollama / vLLM and other OpenAI-compatible options work the same way.
- **Remote platform** (no local models): Alibaba Bailian / OpenRouter etc. — create `.env` in the project root with the address and key (see the example above).

**OCR model (only for scanned documents, Apple Silicon)**: download [Unlimited-OCR-MLX](https://huggingface.co/LoJexLLM/Unlimited-OCR-MLX) to `~/models/Unlimited-OCR-MLX` (commands in "Backend 1"; Sovena loads/releases it on demand).

> **Non-Apple-Silicon computers** doing OCR: use the "GGUF backend" instead — download [Unlimited-OCR-GGUF](https://huggingface.co/sahilchachra/Unlimited-OCR-GGUF) + run llama-server; see "Backend 2".
> Just want a quick taste and no retrieval yet? Model services can be added later — skip ahead to Steps 4–5.

### Step 4: Get Sovena and install dependencies

```bash
git clone https://github.com/<you>/sovena.git
cd sovena
uv sync        # downloads all dependencies (~1.3GB the first time, takes a few minutes)
```

> On non-Apple-Silicon machines (Windows / Linux / Intel Macs): MLX-related dependencies are only used by the MLX OCR backend — `uv sync` automatically skips or installs CPU-compatible versions on these platforms; text-PDF and non-PDF conversion, retrieval, and all other core features, plus OCR via the GGUF backend, all work normally.

### Step 5: One-command startup

```bash
uv run sovena
```

Seeing `Uvicorn running on http://0.0.0.0:8765` means success. Open **http://localhost:8765** in a browser:

- The status dot on the "Monitoring" page is green → the service is healthy
- The collection dropdown on the "Zotero flow" page shows your collections → Zotero is connected

Stop with **Control + C** in the terminal. The service **does not run persistently or auto-start**; heavy operations are internally serialized (OCR concurrency = 1, memory guard) so your machine never gets buried.

### Step 6 (optional): personal path configuration

By default data lives in `~/sovena_data`. To store it elsewhere (e.g. an external drive), create a `.env` in the project root:

```bash
echo 'SOVENA_ROOT=/Volumes/your-disk/sovena_data' > .env
```

`.env` is git-ignored — personal paths never enter the repository.

### Step 7: Run your first job

On the web console: "Zotero flow" → pick a **small collection** (e.g. 5 items) → "Prepare" → watch progress on "Monitoring". When done, ask a question in "Semantic search".

### Troubleshooting

| Symptom | Fix |
| --- | --- |
| `uv: command not found` | Step 1 didn't finish, or the terminal wasn't reopened |
| Port already in use | An old service is still running: `lsof -ti tcp:8765 \| xargs kill`, then restart |
| "Zotero connection failed" | Zotero isn't open, or it's an old version (needs 7.0+) |
| Search errors / no results | Embedding service not running / wrong key (local mlx-lm or a remote platform), or that collection was never prepared; switching embedding models requires rebuilding the index |
| OCR model errors | MLX backend: `Unlimited-OCR-MLX` not downloaded or wrong path; http backend: llama-server not started or `SOVENA_OCR_API` misconfigured (see above) |
| Fans spinning hard | Normal: OCR is heavy; the model unloads automatically when the job finishes |

### Step 8 (optional): connect AI clients (works from other computers too)

Any MCP streamable-http client (Claude Desktop, Cherry Studio, Trae, etc.) — just fill in:

```json
{
  "mcpServers": {
    "sovena": { "url": "http://localhost:8765/mcp" }
  }
}
```

For remote (other-computer) access: start the server with `SOVENA_HOST=0.0.0.0` (the default) and change the client URL to `http://<server-IP-or-Tailscale-hostname>:8765/mcp`. The web console's "Settings" page can copy/download the current deployment's JSON config.

## Zotero Plugin (optional)

`dist/sovena-plugin-<version>.xpi` is a client plugin installable in **Zotero 7+ (incl. 9/10)**, bringing Sovena right into Zotero:

- **Right-click a collection** → "Sovena: prepare/update semantic packs (incremental)"
- **Right-click items** → "Sovena: add attachments to an ad-hoc semantic pack" (local file attachments of the selected items go through the adhoc flow)
- **Tools menu → Sovena** → open the console / set the server address / copy MCP client config / connection check

Install: Zotero → Tools → Plugins → gear icon → Install Plugin From File… → pick `dist/sovena-plugin-0.1.0.xpi`. Defaults to `http://localhost:8765`; **on other computers**, set the Sovena server address under "Tools → Sovena → Server address…".

Rebuild the plugin (after modifying `zotero-plugin/`):

```bash
bash zotero-plugin/build.sh    # produces dist/sovena-plugin-<version>.xpi
```

## Environment Variables

All configuration can be set via environment variables; **recommended**: a `.env` file in the project root (git-ignored, ideal for personal paths), loaded automatically at startup:

```dotenv
SOVENA_ROOT=/Volumes/your-disk/zotero_AI
SOVENA_ZOTERO_API=http://localhost:23119/api
```

| Variable | Default | Description |
| --- | --- | --- |
| `SOVENA_ROOT` | `~/sovena_data` | Root directory of semantic literature packs (**recommended to set in `.env` for personal deployments**; LanceDB lives under it by default) |
| `SOVENA_ZOTERO_API` | `http://localhost:23119/api` | Zotero local API |
| `SOVENA_HOST` | `0.0.0.0` | Service listen address |
| `SOVENA_PORT` | `8765` | Service port |
| `SOVENA_EMBED_API` | `http://localhost:8080/v1` | Embedding service address (local mlx-lm/Ollama or a remote platform) |
| `SOVENA_EMBED_API_KEY` | (empty) | Embedding service key (required for remote commercial platforms) |
| `SOVENA_EMBED_MODEL` | `text-embedding-qwen3-embedding-4b` | Embedding model name |
| `SOVENA_LANCEDB` | `$SOVENA_ROOT/_lancedb` | Vector store directory |
| `SOVENA_OCR_BACKEND` | `auto` | OCR backend: `mlx` / `http` / `auto` (setting `SOVENA_OCR_API` implies http) |
| `SOVENA_OCR_MODEL` | `~/models/Unlimited-OCR-MLX` | MLX backend model directory |
| `SOVENA_OCR_API` | (empty) | http backend service address (e.g. `http://127.0.0.1:8080/v1`) |
| `SOVENA_OCR_MODEL_NAME` | `Unlimited-OCR` | http backend model name |
| `SOVENA_OCR_API_KEY` | (empty) | http backend auth key (if any) |
| `SOVENA_MEM_GUARD_MB` | `12288` | Memory-guard threshold (MB); new jobs are postponed below it |

## Usage

### Zotero literature flow (incremental)

Pick a collection on the web console → "Prepare"; or have an AI client call the MCP tool `sovena_prepare`. Repeated runs are automatically incremental: only new/changed items are processed (Zotero version change or attachment mtime change), with item-level index add/remove. Tick "full rebuild" to force a redo.

### Ad-hoc materials (any files/folders)

Turn e-book libraries, loose PDFs, lecture notes, etc. into searchable semantic packs:

- **Web console**: fill paths in the "Ad-hoc packs" card (separate multiple entries with newlines or `;`) → submit; afterwards they are searchable alongside Zotero collections
- **MCP**: `sovena_adhoc_process(paths=["/path/to/E_book/subdir"], name="my library")`
- **REST**: `POST /api/adhoc/submit` `{"paths": [...], "name": "..."}`

Supports pdf/epub/docx/html/txt/md/xlsx/pptx etc.; scanned PDFs automatically go through OCR; incremental too (unchanged source-file mtime → skipped).

### MCP tools

| Group | Tools |
| --- | --- |
| Zotero reading | `zotero_collections` `zotero_search` `zotero_item` `zotero_annotations` |
| Literature flow | `sovena_prepare` `sovena_manifest` `sovena_read_item` `sovena_search` `sovena_find_similar` |
| Ad-hoc | `sovena_adhoc_process` `sovena_adhoc_list` |
| Jobs/ops | `sovena_job_status` `sovena_jobs` `sovena_cancel_job` `sovena_system_status` `sovena_doctor` |

> Note: the Zotero local API is read-only, so no write tools are provided.

## Semantic pack directory layout

```
$SOVENA_ROOT/
  <collection-name>/
    _manifest.json                 # collection-level manifest (incremental basis)
    <author>_<year>_<title-slug>/
      meta.json                    # Zotero metadata + conversion stats
      content.md                   # AI-friendly markdown (with printed-page-number markers)
  adhoc/
    <pack-name>/
      _manifest.json
      <file-name-slug>/
        meta.json
        content.md
  _lancedb/                        # vector store
```

## REST API overview

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/collections` | Zotero collections + ad-hoc packs and their status |
| GET | `/api/config`, `/api/system` | runtime config; system status (memory/CPU/disk/jobs) |
| GET | `/api/fs/list?path=` | server-side directory listing (for the web path picker) |
| POST | `/api/jobs` | submit a prepare job (collection/limit/use_ocr/rebuild) |
| POST | `/api/adhoc/submit` | submit an ad-hoc job (paths/name/use_ocr/recursive) |
| GET | `/api/jobs[/{id}]` | job list/detail (with logs); POST `/{id}/cancel` to cancel |
| GET | `/api/search?q=` | semantic search (can be scoped to a collection) |
| GET | `/api/manifest/{collection}`, `/api/adhoc/manifest/{name}` | manifests |
| GET | `/api/item/{collection}/{dir}/content` etc. | content/metadata |
| GET | `/api/system`, `/api/mcp-config` | system status; MCP client config |

## Project layout

```
sovena/
  main.py                  # one-command startup entry
  sovena/
    server.py              # service entry (Web + MCP in one process, auto-loads .env)
    web.py / webui.html    # web console (tabbed sections + path picker)
    mcp_server.py          # MCP toolset
    zotero_collector.py    # Zotero local API collection (4-way attachment resolution)
    pipeline.py            # prepare pipeline (incremental)
    adhoc.py               # ad-hoc material processing
    converter.py           # L1/L2/anydoc conversion
    indexer.py             # chunking + embedding + LanceDB
    packager.py            # semantic pack writing
    jobs.py                # job scheduling (memory guard / OCR concurrency = 1)
  zotero-plugin/           # Zotero client plugin source (bootstrap structure)
  dist/                    # build artifacts (sovena-plugin-<version>.xpi)
  ocr_port/                # Unlimited-OCR-MLX (MLX OCR engine)
  .env                     # local personal config (optional, not tracked)
```

## Acknowledgements

Sovena stands on the shoulders of these projects, with deep gratitude:

- **[Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)** (Baidu, MIT) — the document-OCR model itself; `ocr_port/` is ported from the [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) community's MLX implementation; the MLX weights (LoJexLLM) and the GGUF quantization ([sahilchachra](https://huggingface.co/sahilchachra/Unlimited-OCR-GGUF)) both come from the HuggingFace community; the http backend runs via [llama.cpp](https://github.com/ggml-org/llama.cpp)'s llama-server (MIT)
- **[cookjohn/zotero-mcp](https://github.com/cookjohn/zotero-mcp)** — one of the project's sources of inspiration
- **[Zotero](https://www.zotero.org/)** (AGPL) — the literature manager itself and its local API
- **[PyMuPDF](https://github.com/pymupdf/PyMuPDF)** (AGPL) — PDF text extraction and page labels
- **[LanceDB](https://github.com/lancedb/lancedb)** (Apache-2.0) — the local vector store
- **[FastMCP](https://github.com/jlowin/fastmcp)** (MIT) — the MCP service framework
- **[anydoc](https://pypi.org/project/firecrawl-anydoc/)** (firecrawl-anydoc) — non-PDF document conversion (docx/epub etc.)
- **[trafilatura](https://github.com/adbar/trafilatura)** (Apache-2.0) — web main-content extraction
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)** (Apple ml-explore, MIT) — local embedding inference server (`mlx_lm.server`, OpenAI-compatible API)
- **[uv](https://docs.astral.sh/uv/)** (Astral, MIT) — Python package management

## License

MIT © Sovena contributors, Institute of Art Anthropology (Southwest University), Institute of Chinese Music Mental Health (Southwest University); author: Fengkai Shi (sfklc@hotmail.com)

[中文版 README](README.md)

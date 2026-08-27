/* sovena Zotero 插件 bootstrap（Zotero 7+，含 9/10）。
 *
 * 功能：
 *   1. 分类右键 → 「sovena：准备/更新语义包」（POST /api/jobs，服务端增量处理）
 *   2. 条目右键 → 「sovena：把附件加入临时语义包」（POST /api/adhoc/submit）
 *   3. 工具菜单 → 监管台 / 服务器设置 / 复制 MCP 客户端配置 / 连接检查
 *
 * 服务端地址存于 pref extensions.sovena.serverUrl（默认 http://localhost:8765），
 * 换机器部署时仅需改此地址。
 */
/* global Zotero, ChromeUtils, Services */

var serverPref = "extensions.sovena.serverUrl";
var _shutdownFn = null;

var Services = globalThis.Services;
if (!Services) {
  // Zotero 7+ bootstrap 环境没有全局 Services 时
  Services = ChromeUtils.importESModule(
    "resource://gre/modules/Services.sys.mjs"
  ).Services;
}

// ---------------------------------------------------------------------------
// 服务端地址与 HTTP
// ---------------------------------------------------------------------------

function getServerUrl() {
  let url = Services.prefs.getStringPref(serverPref, "http://localhost:8765");
  return url.replace(/\/+$/, "");
}

function setServerUrl(url) {
  Services.prefs.setStringPref(serverPref, url.replace(/\/+$/, ""));
}

/**
 * 调 sovena REST API。
 * @param {string} method GET/POST
 * @param {string} path 如 /api/system
 * @param {object|null} body POST 的 JSON 对象
 * @returns {Promise<object>} 解析后的 JSON；失败抛 Error(message)
 */
async function api(method, path, body) {
  const url = getServerUrl() + path;
  const options = {
    responseType: "json",
    timeout: 15000,
    headers: body !== undefined ? { "Content-Type": "application/json" } : {},
    body: body !== undefined ? JSON.stringify(body) : undefined,
  };
  try {
    const req = await Zotero.HTTP.request(method, url, options);
    if (req.status >= 200 && req.status < 300) {
      return typeof req.response === "object" ? req.response : JSON.parse(req.responseText);
    }
    let detail = req.responseText || "";
    try { detail = JSON.parse(detail).error || detail; } catch (e) { /* 保持原样 */ }
    throw new Error(`HTTP ${req.status}: ${String(detail).slice(0, 200)}`);
  } catch (e) {
    if (e instanceof Zotero.HTTP.TimeoutException || /status 0|NetworkError/i.test(String(e))) {
      throw new Error(`无法连接 sovena 服务端 ${getServerUrl()}（服务是否已启动？地址可在 工具→sovena 中修改）`);
    }
    throw e;
  }
}

// ---------------------------------------------------------------------------
// 通知
// ---------------------------------------------------------------------------

function notify(headline, desc) {
  try {
    const pw = new Zotero.ProgressWindow({ closeOnClick: true });
    pw.changeHeadline(headline);
    pw.addDescription(desc);
    pw.show();
    pw.startCloseTimer(6000);
  } catch (e) {
    Zotero.debug(`sovena: 通知失败 ${e}`);
  }
}

function alertError(win, e) {
  const msg = e && e.message ? e.message : String(e);
  Zotero.logError(e);
  Services.prompt.alert(win, "sovena 错误", msg);
}

// ---------------------------------------------------------------------------
// 动作
// ---------------------------------------------------------------------------

async function prepareCollection(win) {
  try {
    const treeRow = win.ZoteroPane.getCollectionTreeRow();
    const ref = treeRow && treeRow.ref;
    if (!ref || !ref.name) {
      notify("sovena", "请先在左侧选中一个分类再操作。");
      return;
    }
    // 判断是库还是分类
    const isLibrary = ref.isLibrary === true || ref.libraryID !== undefined && ref.name === undefined;
    if (isLibrary) {
      notify("sovena", "请选择具体分类（不支持整个文库）。");
      return;
    }
    const job = await api("POST", "/api/jobs", {
      collection: ref.name,
      use_ocr: true,
      rebuild: false,
    });
    notify("sovena", `已提交「${ref.name}」语义包准备任务（${job.id}）`,
      "增量处理：仅新增/变更条目会重新转换。进度见 sovena 监管台。");
  } catch (e) {
    alertError(win, e);
  }
}

async function addItemsToAdhoc(win) {
  try {
    const items = win.ZoteroPane.getSelectedItems();
    const regular = items.filter((it) => it && !it.isAttachment && !it.isNote && !it.isFeedItem);
    if (!regular.length) {
      notify("sovena", "请先选中文献条目（右键其附件将被处理）。");
      return;
    }
    // 汇总所有条目的文件附件路径
    const paths = [];
    for (const item of regular) {
      if (item.isRegularItem() !== true && !item.isRegularItem) continue;
      let attIds = [];
      if (typeof item.getAttachments === "function") {
        attIds = item.getAttachments() /* 同步：主库 */ || [];
      }
      for (const id of attIds) {
        const att = Zotero.Items.get(id);
        if (!att || !att.isAttachment()) continue;
        let path = null;
        try { path = await att.getFilePathAsync(); } catch (e) { path = null; }
        if (!path) continue;
        paths.push(path);
      }
    }
    if (!paths.length) {
      notify("sovena", "所选条目没有本地文件附件（快照/URL 附件无法作为临时包处理）。");
      return;
    }
    const first = regular[0];
    const name = `Zotero选取-${first.getDisplayTitle().slice(0, 40)}`;
    const job = await api("POST", "/api/adhoc/submit", {
      paths,
      name,
      use_ocr: true,
      recursive: false,
      index: true,
    });
    notify("sovena", `已提交临时语义包「${name}」（${paths.length} 个文件，任务 ${job.id}）`,
      "扫描版 PDF 将自动走 OCR。进度见 sovena 监管台。");
  } catch (e) {
    alertError(win, e);
  }
}

async function openDashboard() {
  try {
    Zotero.launchURL(getServerUrl() + "/");
  } catch (e) {
    Zotero.logError(e);
  }
}

function configureServer(win) {
  const input = { value: getServerUrl() };
  const ok = Services.prompt.prompt(
    win,
    "sovena 服务端地址",
    "sovena 服务端地址（本机或远程机器，如 http://192.168.1.10:8765）：",
    input,
    null,
    {}
  );
  if (ok && input.value.trim()) {
    setServerUrl(input.value.trim());
    notify("sovena", `服务端地址已保存：${getServerUrl()}`);
  }
}

async function copyMcpConfig(win) {
  try {
    const cfg = await api("GET", "/api/mcp-config");
    const text = JSON.stringify(cfg, null, 2);
    await Zotero.Utilities.Internal.copyTextToClipboard(text);
    notify("sovena", "MCP 客户端配置已复制到剪贴板", text.replace(/\s+/g, " ").slice(0, 120));
  } catch (e) {
    alertError(win, e);
  }
}

async function checkConnection(win) {
  try {
    const st = await api("GET", "/api/system");
    notify("sovena", `连接正常（${getServerUrl()}）`,
      `内存可用 ${(st.mem_available_mb / 1024).toFixed(1)}GB / 守卫阈值 ${(st.mem_guard_mb / 1024).toFixed(0)}GB，CPU ${st.cpu_percent}%`);
  } catch (e) {
    alertError(win, e);
  }
}

// ---------------------------------------------------------------------------
// 菜单注入
// ---------------------------------------------------------------------------

function onMainWindowLoad(win) {
  const doc = win.document;
  if (doc.getElementById("sovena-collection-menuitem")) return; // 幂等

  // ---- 分类右键菜单 ----
  const collMenu =
    doc.getElementById("zotero-collectionmenu") ||
    doc.getElementById("collectionContextMenu");
  if (collMenu) {
    const mi = doc.createXULElement("menuitem");
    mi.id = "sovena-collection-menuitem";
    mi.setAttribute("label", "准备/更新语义包（增量）");
    mi.addEventListener("command", () => prepareCollection(win));
    collMenu.appendChild(mi);
  }

  // ---- 条目右键菜单 ----
  const itemMenu = doc.getElementById("zotero-itemmenu");
  if (itemMenu) {
    const mi = doc.createXULElement("menuitem");
    mi.id = "sovena-item-menuitem";
    mi.setAttribute("label", "把附件加入临时语义包");
    mi.addEventListener("command", () => addItemsToAdhoc(win));
    itemMenu.appendChild(mi);
  }

  // ---- 工具菜单 ----
  const toolsPopup = doc.getElementById("menu_ToolsPopup");
  if (toolsPopup) {
    const menu = doc.createXULElement("menu");
    menu.id = "sovena-tools-menu";
    menu.setAttribute("label", "缙云文采");
    const popup = doc.createXULElement("menupopup");
    popup.id = "sovena-tools-popup";

    const mk = (label, fn) => {
      const it = doc.createXULElement("menuitem");
      it.setAttribute("label", label);
      it.addEventListener("command", fn);
      return it;
    };
    popup.appendChild(mk("打开 sovena 监管台…", () => openDashboard()));
    popup.appendChild(mk("服务端地址…", () => configureServer(win)));
    popup.appendChild(mk("复制 MCP 客户端配置", () => copyMcpConfig(win)));
    popup.appendChild(mk("检查 sovena 连接", () => checkConnection(win)));
    // 兜底入口（右键菜单 ID 变化时功能仍可用）
    popup.appendChild(doc.createXULElement("menuseparator"));
    popup.appendChild(
      mk("准备当前分类语义包", () => prepareCollection(win))
    );
    popup.appendChild(
      mk("把选中条目附件加入临时语义包", () => addItemsToAdhoc(win))
    );

    menu.appendChild(popup);
    toolsPopup.appendChild(menu);
  }
}

function onMainWindowUnload(win) {
  const doc = win.document;
  for (const id of [
    "sovena-collection-menuitem",
    "sovena-item-menuitem",
    "sovena-tools-menu",
  ]) {
    const el = doc.getElementById(id);
    if (el) el.remove();
  }
}

// ---------------------------------------------------------------------------
// 插件生命周期（Zotero 7+ bootstrap 约定）
// ---------------------------------------------------------------------------

function install() {}

function uninstall() {}

async function startup({ rootURI }) {
  await Zotero.initializationPromise;

  const loadedWindows = [];
  const windowListener = {
    onOpenWindow: (aWindow) => {
      const domWindow = aWindow.docShell
        ? aWindow.docShell.domWindow
        : aWindow.QueryInterface(Ci.nsIInterfaceRequestor).getInterface(Ci.nsIDOMWindow);
      domWindow.addEventListener(
        "load",
        () => {
          if (domWindow.ZoteroPane) {
            loadedWindows.push(domWindow);
            onMainWindowLoad(domWindow);
          }
        },
        { once: true }
      );
    },
  };

  Services.wm.addListener(windowListener);

  // 已打开的主窗口
  for (const win of Zotero.getMainWindows()) {
    if (win.ZoteroPane) {
      loadedWindows.push(win);
      onMainWindowLoad(win);
    }
  }

  _shutdownFn = () => {
    Services.wm.removeListener(windowListener);
    for (const win of loadedWindows) onMainWindowUnload(win);
  };
}

function shutdown() {
  if (_shutdownFn) _shutdownFn();
  _shutdownFn = null;
}

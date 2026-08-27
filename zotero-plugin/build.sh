#!/usr/bin/env bash
# 打包 Zotero 插件为 .xpi（Zotero 7+/9/10 直接安装）
# 用法: bash zotero-plugin/build.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../dist
rm -f ../dist/litflow-plugin-*.xpi
VERSION=$(python3 -c 'import json;print(json.load(open("manifest.json"))["version"])')
zip -r -X "../dist/litflow-plugin-${VERSION}.xpi" manifest.json bootstrap.js prefs.js -x '*.DS_Store' >/dev/null
echo "已生成 ../dist/litflow-plugin-${VERSION}.xpi"

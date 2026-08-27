"""P0-5 验证：Zotero 采集器（分类树 + 4 路附件解析）。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from litflow import ZoteroCollector

col = ZoteroCollector()
print(f"data_dir     = {col.data_dir}")
print(f"linked_base  = {col.linked_base}")

# 1. 分类解析（含子分类）
coll = col.resolve_collection("古琴研究")
print(f"\n解析分类: {coll['name']} key={coll['key']} items={coll['num_items']}")
subs = col.subcollection_keys(coll["key"])
print(f"子分类 {len(subs)} 个: {subs}")

# 2. 收集条目 + 附件
records = col.collect("古琴研究", recursive=True)
print(f"\n条目数: {len(records)}")

stats = {"records": len(records), "with_pdf": 0, "pdf_exists": 0,
         "snapshot": 0, "linked_url": 0, "missing": 0, "no_attach": 0}
for r in records:
    pdfs = [a for a in r.attachments if a.is_pdf]
    if pdfs:
        stats["with_pdf"] += 1
        stats["pdf_exists"] += sum(a.exists for a in pdfs)
    stats["snapshot"] += sum(1 for a in r.attachments if a.is_snapshot and a.link_mode != "linked_url")
    stats["linked_url"] += sum(1 for a in r.attachments if a.link_mode == "linked_url")
    stats["missing"] += sum(1 for a in r.attachments if a.path and not a.exists)
    if not r.attachments:
        stats["no_attach"] += 1

print("统计:", stats)

# 3. 抽样展示
print("\n样例条目:")
shown = 0
for r in records:
    if r.attachments and shown < 5:
        at = r.attachments[0]
        flag = "OK" if at.exists else ("URL" if at.url else "MISS")
        print(f"  [{flag}] {r.creator_summary()} {r.year} 《{r.title[:38]}》"
              f" | {at.content_type} | {at.path or at.url}")
        shown += 1

# 4. 分类树片段
tree = col.collection_tree()
print(f"\n分类树根节点数: {len(tree)}")

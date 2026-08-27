"""缙云文采（Sovena）：学科学术文献流处理机制（各学科通用）。

「缙云」取自北碚缙云山（西南大学所在地）；「文采」兼含二义：
既是采撷文献之精华，亦是文章之华采。
"""

from .zotero_collector import Attachment, LitRecord, ZoteroCollector, ZoteroPrefs

__all__ = ["Attachment", "LitRecord", "ZoteroCollector", "ZoteroPrefs"]

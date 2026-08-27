"""缙云文献流（jinyun）：学科学术文献流处理机制（各学科通用）。

「缙云」取自北碚缙云山（西南大学所在地）；「文献流」言其功能：
让文献如活水，流经检索、流经 AI、流进你的研究。
"""

from .zotero_collector import Attachment, LitRecord, ZoteroCollector, ZoteroPrefs

__all__ = ["Attachment", "LitRecord", "ZoteroCollector", "ZoteroPrefs"]

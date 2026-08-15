"""检索入口（薄封装）：委托给 rag 模块的混合检索（Milvus 向量 + BM25）。

Milvus 不可用时 rag.retriever 自动降级为纯 BM25（基于 SQLite 分块文本缓存）。
"""

from __future__ import annotations

from .rag.retriever import SearchHit, search

__all__ = ["SearchHit", "search"]

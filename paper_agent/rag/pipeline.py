"""RAG 流水线：文档解析 → 分块 → 嵌入 → Milvus 建索引，支持增量、更新、删除与迁移。

- index_paper(paper_id, force=False)：把单篇论文分块嵌入并写入 Milvus
  （增量：已在索引中且未 force 则跳过；force=True 为更新语义：先删后写）
- index_missing()：增量更新——把库中尚未建索引的论文全部入索引
- delete_paper_vectors(paper_id)：删除某论文的向量（Milvus）
- remove_paper(paper_id)：删除论文（Milvus 向量 + SQLite 元数据，级联清理分块/创新点）
- migrate_from_sqlite()：老数据迁移（把 SQLite 里已有分块全部写入 Milvus）
- status()：索引状态摘要（供 UI / MCP 展示）
"""

from __future__ import annotations

import logging

from ..db import get_chunks, get_conn, get_meta_int, get_paper, list_papers, set_meta
from ..embed import get_embedder
from .config import load_rag_config
from .milvus_store import MilvusStore

logger = logging.getLogger("paper_agent")

_store: MilvusStore | None = None


def get_store() -> MilvusStore:
    global _store
    if _store is None:
        _store = MilvusStore()
    return _store


def _bump_version() -> None:
    set_meta("rag_index_version", str(get_meta_int("rag_index_version") + 1))


def index_embedded_chunks(paper_id: int, chunks: list[dict], vecs) -> dict:
    """把已嵌入好的分块写入 Milvus（入库流水线内复用已算向量，避免重复嵌入）。"""
    store = get_store()
    if not store.available:
        return {"ok": False, "error": "Milvus 不可用"}
    existing = store.count_by_paper(paper_id)
    if existing:
        store.delete_by_paper(paper_id)
    rows = [
        {
            "paper_id": paper_id,
            "seq": c["seq"],
            "heading": c.get("heading"),
            "content": c["content"],
            "embedding": v,
        }
        for c, v in zip(chunks, vecs)
    ]
    n = store.insert_chunks(rows)
    _bump_version()
    return {"ok": True, "indexed": n, "message": f"已写入 {n} 块向量索引"}


def index_paper(paper_id: int, force: bool = False) -> dict:
    """解析（读 PDF/分块缓存）→ 分块 → 嵌入 → 写入 Milvus。

    force=False：增量语义，已在索引中则跳过；
    force=True：更新语义，删除旧向量后重建该论文索引。
    """
    store = get_store()
    if not store.available:
        return {"ok": False, "error": "Milvus 不可用，无法建立索引（请检查 configs/rag.yaml 的 milvus.uri）"}
    paper = get_paper(paper_id)
    if not paper:
        return {"ok": False, "error": f"论文 #{paper_id} 不存在"}
    chunks = get_chunks(paper_id)
    if not chunks:
        return {"ok": False, "error": f"《{paper['title'][:40]}》尚未分块，请先入库"}

    existing = store.count_by_paper(paper_id)
    if existing and not force:
        return {"ok": True, "indexed": 0, "message": f"论文已在索引中（{existing} 块），增量跳过"}

    embedder = get_embedder()
    if not embedder.available:
        return {"ok": False, "error": "本地 embedding 模型不可用，无法建立向量索引"}

    vecs = embedder.embed_many([c["content"] for c in chunks])
    if vecs is None:
        return {"ok": False, "error": "嵌入失败"}

    return index_embedded_chunks(paper_id, chunks, vecs)


def index_missing() -> dict:
    """增量更新：把库中尚未建索引（或分块数不一致）的论文全部入索引。"""
    store = get_store()
    if not store.available:
        return {"ok": False, "error": "Milvus 不可用"}
    indexed_ids = set(store.paper_ids())
    results = []
    total = 0
    for paper in list_papers():
        if not get_chunks(paper["id"]):
            continue
        n_chunks = len(get_chunks(paper["id"]))
        if paper["id"] in indexed_ids and store.count_by_paper(paper["id"]) == n_chunks:
            continue  # 已是最新
        r = index_paper(paper["id"], force=True)
        if r.get("ok"):
            total += r.get("indexed", 0)
        results.append({"paper_id": paper["id"], "title": paper["title"][:40], **r})
    _bump_version()
    return {"ok": True, "total_indexed": total, "papers": results}


def delete_paper_vectors(paper_id: int) -> dict:
    """删除某论文在 Milvus 中的全部分块。"""
    store = get_store()
    if not store.available:
        return {"ok": False, "error": "Milvus 不可用"}
    n = store.delete_by_paper(paper_id)
    _bump_version()
    return {"ok": True, "deleted": n, "message": f"已删除论文 #{paper_id} 的 {n} 块向量"}


def remove_paper(paper_id: int) -> dict:
    """完整删除论文：Milvus 向量 + SQLite 元数据（级联清理分块/创新点）。"""
    vectors = delete_paper_vectors(paper_id)
    paper = get_paper(paper_id)
    if paper:
        with get_conn() as conn:
            conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
    _bump_version()
    return {
        "ok": True,
        "paper_id": paper_id,
        "title": paper["title"][:60] if paper else "",
        "vectors_deleted": vectors.get("deleted", 0),
    }


def migrate_from_sqlite() -> dict:
    """老数据迁移：把 SQLite 中已存在的分块全部写入 Milvus（幂等：已存在的跳过）。"""
    return index_missing()


def status() -> dict:
    """RAG 索引状态摘要。"""
    cfg = load_rag_config()
    store = get_store()
    return {
        "ok": True,
        "milvus_uri": cfg.milvus.resolved_uri(),
        "milvus_available": store.available,
        "collection": cfg.milvus.collection,
        "vector_count": store.count() if store.available else 0,
        "indexed_papers": len(store.paper_ids()) if store.available else 0,
        "total_papers": len(list_papers()),
        "embedding_model": cfg.embedding.model,
        "embedding_available": get_embedder().available,
        "chunk_size": cfg.chunking.size,
        "top_k": cfg.retrieval.top_k,
    }

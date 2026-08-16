"""检索：Milvus BM25 倒排索引；embedding 启用时与稠密向量做 RRF 混合融合。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..db import list_papers
from .config import load_rag_config
from .embedding import get_embedder
from .pipeline import get_store

logger = logging.getLogger("paper_agent")

RRF_K = 60  # RRF 融合常数


@dataclass
class SearchHit:
    paper_id: int
    arxiv_id: str
    title: str
    chunk_seq: int
    heading: str | None
    snippet: str
    score: float
    used_vector: bool

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "chunk_seq": self.chunk_seq,
            "heading": self.heading,
            "snippet": self.snippet,
            "score": round(float(self.score), 4),
            "used_vector": self.used_vector,
        }


def search(query: str, top_k: int | None = None) -> list[SearchHit]:
    cfg = load_rag_config()
    top_k = top_k or cfg.retrieval.top_k
    store = get_store()
    if not store.available or store.count() == 0:
        logger.warning("Milvus 不可用或索引为空，检索暂不可用")
        return []

    papers = {p["id"]: p for p in list_papers()}
    candidates = store.search_bm25(query, top_k * cfg.retrieval.candidate_scale)

    used_vector = False
    if cfg.embedding.enabled:
        q = get_embedder().embed_many([query])
        if q is not None:
            dense = store.search_dense(q[0], top_k * cfg.retrieval.candidate_scale)
            if dense:
                used_vector = True
                # RRF 融合：BM25 与向量排名互不影响、量纲无关
                by_key = {(h["paper_id"], h["seq"]): h for h in candidates}
                for h in dense:
                    by_key.setdefault((h["paper_id"], h["seq"]), h)
                fused: dict[tuple[int, int], float] = {}
                for rank, h in enumerate(candidates):
                    fused[(h["paper_id"], h["seq"])] = 1.0 / (RRF_K + rank + 1)
                for rank, h in enumerate(dense):
                    key = (h["paper_id"], h["seq"])
                    fused[key] = fused.get(key, 0) + 1.0 / (RRF_K + rank + 1)
                candidates = [
                    {**by_key[key], "score": s}
                    for key, s in sorted(fused.items(), key=lambda kv: -kv[1])
                ]

    hits: list[SearchHit] = []
    per_paper: dict[int, int] = {}
    for h in candidates:
        if len(hits) >= top_k:
            break
        pid = h["paper_id"]
        if per_paper.get(pid, 0) >= cfg.retrieval.max_per_paper:
            continue  # 多样性上限：每篇论文最多 max_per_paper 条
        per_paper[pid] = per_paper.get(pid, 0) + 1
        paper = papers.get(pid, {})
        hits.append(
            SearchHit(
                paper_id=pid,
                arxiv_id=paper.get("arxiv_id") or "",
                title=paper.get("title", "未知论文"),
                chunk_seq=h["seq"],
                heading=h.get("heading"),
                snippet=h["content"][:500],
                score=h["score"],
                used_vector=used_vector,
            )
        )
    return hits

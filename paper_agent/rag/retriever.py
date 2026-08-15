"""混合检索：Milvus 向量检索 + jieba BM25 加权融合，带每篇论文多样性上限。

Milvus 不可用时自动降级为纯 BM25（基于 SQLite 分块文本缓存）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from ..db import all_chunks, get_meta_int, list_papers
from ..embed import get_embedder
from .config import load_rag_config
from .pipeline import get_store

logger = logging.getLogger("paper_agent")

STOPWORDS = {
    "的", "了", "和", "是", "在", "与", "及", "等", "对", "为", "中", "上", "下",
    "the", "a", "an", "of", "to", "in", "and", "for", "on", "is", "are", "we", "our",
}


def tokenize(text: str) -> list[str]:
    return [w for w in jieba.cut_for_search(text) if w not in STOPWORDS and len(w.strip()) > 1]


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


class HybridRetriever:
    """按 rag_index_version 增量重建 BM25 内存索引；向量检索走 Milvus。"""

    def __init__(self):
        self._version = -1
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []
        self._papers: dict[int, dict] = {}

    def _ensure_bm25(self) -> None:
        version = get_meta_int("rag_index_version")
        if version == self._version and self._bm25 is not None:
            return
        self._papers = {p["id"]: p for p in list_papers()}
        store = get_store()
        self._chunks = store.fetch_all() if store.available else all_chunks()
        docs = [c["content"] for c in self._chunks]
        self._bm25 = BM25Okapi([tokenize(d) for d in docs]) if docs else None
        self._version = version
        logger.info("BM25 索引已重建（version=%d，chunks=%d）", version, len(self._chunks))

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        cfg = load_rag_config()
        top_k = top_k or cfg.retrieval.top_k
        self._ensure_bm25()
        if not self._chunks or self._bm25 is None:
            return []

        bm25_scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
        bm25_norm = _minmax(bm25_scores)

        dense: dict[int, float] = {}
        store = get_store()
        use_vector = store.available
        if use_vector:
            q = get_embedder().embed_many([query])
            if q is None:
                use_vector = False
            else:
                candidates = store.search(
                    q[0], top_k=max(top_k * cfg.retrieval.candidate_scale, 16)
                )
                # 用 seq 建立 chunk 定位键（Milvus 返回 chunk 内容，按 (paper_id, seq) 反查）
                for c in candidates:
                    dense[(c["paper_id"], c["seq"])] = (c["score"] + 1.0) / 2.0

        # 候选集：Milvus 向量命中 ∪ BM25 高分 chunk
        union: list[int] = []
        seen: set[int] = set()
        if dense:
            for key in sorted(dense, key=dense.get, reverse=True):  # type: ignore[arg-type]
                i = self._locate(*key)
                if i is not None and i not in seen:
                    union.append(i)
                    seen.add(i)
        bm25_order = np.argsort(bm25_scores)[::-1][: top_k * 4]
        for i in bm25_order:
            if i not in seen:
                union.append(i)
                seen.add(i)

        w = cfg.retrieval.bm25_weight
        scored: list[tuple[float, int]] = []
        for i in union:
            chunk = self._chunks[i]
            key = (chunk["paper_id"], chunk["seq"])
            if use_vector and key in dense:
                score = w * bm25_norm[i] + (1 - w) * dense[key]
            else:
                score = bm25_norm[i]  # 无向量命中时按 BM25 打分
            if score <= 0 and bm25_scores[i] <= 0:
                continue
            scored.append((float(score), i))
        scored.sort(key=lambda x: -x[0])

        # 多样性上限：每篇论文最多 max_per_paper 条
        hits: list[SearchHit] = []
        per_paper: dict[int, int] = {}
        for score, i in scored:
            if len(hits) >= top_k:
                break
            chunk = self._chunks[i]
            pid = chunk["paper_id"]
            if per_paper.get(pid, 0) >= cfg.retrieval.max_per_paper:
                continue
            per_paper[pid] = per_paper.get(pid, 0) + 1
            paper = self._papers.get(pid, {})
            hits.append(
                SearchHit(
                    paper_id=pid,
                    arxiv_id=paper.get("arxiv_id") or "",
                    title=paper.get("title", "未知论文"),
                    chunk_seq=chunk["seq"],
                    heading=chunk.get("heading"),
                    snippet=chunk["content"][:500],
                    score=score,
                    used_vector=use_vector,
                )
            )
        return hits

    def _locate(self, paper_id: int, seq: int) -> int | None:
        for i, c in enumerate(self._chunks):
            if c["paper_id"] == paper_id and c["seq"] == seq:
                return i
        return None


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - lo) / (hi - lo)


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def search(query: str, top_k: int | None = None) -> list[SearchHit]:
    """RAG 检索入口。"""
    return get_retriever().search(query, top_k)

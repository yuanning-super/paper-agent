"""入库工作流（LangGraph StateGraph）：arxiv 论文 → 元数据 → PDF → 文本 → 分块 → 嵌入 → 增强 → 索引。"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from ..db import (
    find_paper_by_arxiv_id,
    get_chunks,
    get_meta_int,
    insert_paper,
    replace_chunks,
    set_meta,
    update_paper,
)
from ..embed import get_embedder
from ..ingestion import (
    IngestResult,
    chunk_text,
    download_pdf,
    enrich_metadata,
    extract_text,
    fetch_arxiv_metadata,
    normalize_arxiv_id,
)


class IngestState(TypedDict, total=False):
    arxiv_id: str
    pdf_url: str
    pdf_path: str
    paper_id: int
    is_new: bool
    title: str
    text: str
    is_scanned: bool
    chunk_count: int
    status: str
    error: str
    # 进度消息（跨节点累加，供 UI 展示）
    events: Annotated[list[str], operator.add]


# ---------- 节点 ----------

def fetch_node(state: IngestState) -> dict:
    try:
        meta = fetch_arxiv_metadata(state["arxiv_id"])
        out: dict = {
            "pdf_url": meta["pdf_url"],
            "title": meta["title"],
            "events": [f"已获取 arXiv 元数据：{meta['title']}"],
        }
        existing = find_paper_by_arxiv_id(meta["arxiv_id"])
        if existing:
            out.update(
                paper_id=existing["id"],
                is_new=False,
                status=existing["status"],
                chunk_count=len(get_chunks(existing["id"])),
            )
            out["events"].append(f"论文已在库中（#{existing['id']}），跳过后续步骤")
            return out
        paper_id, _ = insert_paper({**meta, "source": "arxiv"})
        if meta.get("github_hint"):
            update_paper(paper_id, github_url=meta["github_hint"])
        out.update(paper_id=paper_id, is_new=True, status="metadata")
        out["events"].append(f"论文已登记（#{paper_id}）")
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": f"获取 arXiv 元数据失败：{e}"}


def download_node(state: IngestState) -> dict:
    try:
        pdf_path = download_pdf(state["arxiv_id"], state["pdf_url"])
        update_paper(state["paper_id"], pdf_path=str(pdf_path))
        return {"pdf_path": str(pdf_path), "events": [f"PDF 已下载：{pdf_path.name}"]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"PDF 下载失败：{e}"}


def extract_node(state: IngestState) -> dict:
    from pathlib import Path

    try:
        text, is_scanned = extract_text(Path(state["pdf_path"]))
        if is_scanned:
            update_paper(state["paper_id"], status="text")
            return {
                "is_scanned": True,
                "status": "text",
                "error": "扫描版 PDF：未提取到文本内容，建议上传文本版 PDF 或启用 OCR。",
                "events": ["扫描版 PDF：未提取到文本内容"],
            }
        update_paper(state["paper_id"], status="text")
        return {
            "text": text,
            "is_scanned": False,
            "status": "text",
            "events": [f"文本提取完成：{len(text)} 字符"],
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"PDF 文本提取失败：{e}"}


def chunk_node(state: IngestState) -> dict:
    chunks = chunk_text(state["text"])
    if not chunks:
        return {"error": "分块结果为空"}
    replace_chunks(state["paper_id"], chunks)
    update_paper(state["paper_id"], status="chunked")
    return {
        "chunk_count": len(chunks),
        "status": "chunked",
        "events": [f"分块完成：{len(chunks)} 块"],
    }


def embed_node(state: IngestState) -> dict:
    embedder = get_embedder()
    if not embedder.available:
        return {"events": ["embedding 模型不可用，跳过向量化（检索将用纯 BM25）"]}
    chunks = get_chunks(state["paper_id"])
    vecs = embedder.embed_many([c["content"] for c in chunks])
    if vecs is None:
        return {"events": ["embedding 失败，跳过向量化（检索将用纯 BM25）"]}
    for c, v in zip(chunks, vecs):
        c["embedding"] = v.tobytes()
    from ..db import Chunk

    replace_chunks(
        state["paper_id"],
        [
            Chunk(
                paper_id=state["paper_id"],
                seq=c["seq"],
                heading=c["heading"],
                content=c["content"],
                embedding=c["embedding"],
            )
            for c in chunks
        ],
    )
    update_paper(state["paper_id"], status="embedded")

    # 写入 Milvus 向量索引（复用已算向量；Milvus 不可用则降级纯 BM25，不阻塞入库）
    from ..rag.pipeline import index_embedded_chunks

    r = index_embedded_chunks(state["paper_id"], chunks, vecs)
    if r.get("ok"):
        return {"status": "embedded", "events": [f"向量嵌入完成，已写入 Milvus 索引（{r['indexed']} 块）"]}
    return {"status": "embedded", "events": [f"向量嵌入完成；Milvus 索引失败（{r.get('error')}），检索将降级纯 BM25"]}


def enrich_node(state: IngestState) -> dict:
    try:
        if enrich_metadata(state["paper_id"]):
            return {"events": ["元数据增强完成（中文标题/关键词/分类）"]}
        return {"events": ["元数据增强未执行（无摘要或 LLM 调用失败）"]}
    except Exception as e:  # noqa: BLE001 —— 增强失败不阻塞入库
        return {"events": [f"元数据增强失败（{e}），保留原始元数据"]}


def index_node(state: IngestState) -> dict:
    update_paper(state["paper_id"], status="enriched")
    set_meta("index_version", str(get_meta_int("index_version") + 1))
    return {"status": "enriched", "events": ["入库完成 ✓"]}


# ---------- 路由 ----------

def _router_after_fetch(state: IngestState) -> str:
    if state.get("error"):
        return END
    if not state.get("is_new", True):
        return END  # 已入库，幂等返回
    return "download"


def _make_router(next_node: str):
    def router(state: IngestState) -> str:
        return END if state.get("error") else next_node

    return router


def build_ingest_graph(checkpointer=None):
    g = StateGraph(IngestState)
    g.add_node("fetch", fetch_node)
    g.add_node("download", download_node)
    g.add_node("extract", extract_node)
    g.add_node("chunk", chunk_node)
    g.add_node("embed", embed_node)
    g.add_node("enrich", enrich_node)
    g.add_node("index", index_node)

    g.add_edge(START, "fetch")
    g.add_conditional_edges("fetch", _router_after_fetch, {"download": "download", END: END})
    g.add_conditional_edges("download", _make_router("extract"), {"extract": "extract", END: END})
    g.add_conditional_edges("extract", _make_router("chunk"), {"chunk": "chunk", END: END})
    g.add_conditional_edges("chunk", _make_router("embed"), {"embed": "embed", END: END})
    g.add_conditional_edges("embed", _make_router("enrich"), {"enrich": "enrich", END: END})
    g.add_conditional_edges("enrich", _make_router("index"), {"index": "index", END: END})
    g.add_edge("index", END)
    return g.compile(checkpointer=checkpointer)


def run_ingest(arxiv_id: str, checkpointer=None) -> IngestResult:
    """执行入库工作流并转换为 IngestResult。"""
    graph = build_ingest_graph(checkpointer)
    arxiv_id = normalize_arxiv_id(arxiv_id)
    state = graph.invoke({"arxiv_id": arxiv_id})
    return IngestResult(
        paper_id=state.get("paper_id"),
        is_new=state.get("is_new", False),
        title=state.get("title", ""),
        status=state.get("status", "metadata"),
        is_scanned=state.get("is_scanned", False),
        chunk_count=state.get("chunk_count", 0),
        error=state.get("error"),
        events=state.get("events", []),
    )

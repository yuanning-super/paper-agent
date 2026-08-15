"""RAG MCP 工具服务器（mcp SDK 2.x / MCPServer）。

把论文库 RAG 能力暴露为 MCP 工具：检索、入库、增量更新、删除、状态。

配置（configs/rag.yaml 的 mcp 段）：
  transport: stdio —— Claude Code 注册：
      claude mcp add paper-rag -- uv run python -m paper_agent.rag.mcp_server
  transport: sse   —— 供其他 MCP 客户端通过 HTTP 访问（host/port 可配）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.mcpserver import MCPServer

from ..db import find_paper_by_arxiv_id, list_papers
from ..graphs import run_ingest
from ..ingestion import normalize_arxiv_id
from .config import load_rag_config
from .pipeline import (
    delete_paper_vectors,
    get_store,
    index_missing,
    index_paper,
    remove_paper,
    status,
)
from .retriever import search

logger = logging.getLogger("paper_agent")

_cfg = load_rag_config()
mcp = MCPServer(
    name=_cfg.mcp.name,
    version="0.1.0",
    instructions=(
        "论文库 RAG 工具：基于 Milvus 向量库 + BM25 的混合检索，"
        "支持论文入库（自动解析/分块/建索引）、增量更新与删除。"
    ),
)


def _dump(data: Any, limit: int = 8000) -> str:
    text = json.dumps(data, ensure_ascii=False)
    if len(text) > limit:
        text = text[:limit] + f"…（JSON 过长已截断，共 {len(text)} 字符）"
    return text


@mcp.tool()
def rag_search(query: str, top_k: int = 5) -> str:
    """在论文库中混合检索（Milvus 向量 + BM25）。返回最相关的论文片段，
    包含 paper_id、标题、分块序号、片段内容与相关度得分。"""
    hits = search(query, top_k=top_k)
    return _dump({"ok": True, "count": len(hits), "hits": [h.to_dict() for h in hits]})


@mcp.tool()
def rag_add_paper(source: str) -> str:
    """把论文入库并建立向量索引。source 可以是：
    - arXiv ID（如 1706.03762 或 arXiv:2205.14135v2）：下载 PDF、解析、分块、嵌入、写入 Milvus；
    - 库内论文 ID（如 5）：为已入库论文（重新）建立索引。
    文档解析、分块与建索引全部由 RAG 流水线完成。"""
    source = source.strip()
    try:
        if source.isdigit():
            return _dump(index_paper(int(source), force=True))
        arxiv_id = normalize_arxiv_id(source)
        existing = find_paper_by_arxiv_id(arxiv_id)
        if existing:
            return _dump(index_paper(existing["id"], force=True))
        result = run_ingest(arxiv_id)
        if result.error:
            return _dump({"ok": False, "error": result.error, "events": result.events})
        return _dump(
            {"ok": True, "paper_id": result.paper_id, "title": result.title, "events": result.events}
        )
    except Exception as e:  # noqa: BLE001
        return _dump({"ok": False, "error": f"入库失败：{e}"})


@mcp.tool()
def rag_update_paper(paper_id: int) -> str:
    """更新论文索引：删除旧向量后重新解析、分块、嵌入、写入 Milvus。"""
    return _dump(index_paper(paper_id, force=True))


@mcp.tool()
def rag_delete_paper(paper_id: int, remove_library: bool = True) -> str:
    """删除论文。remove_library=true 时同时从论文库（元数据/分块）与 Milvus 向量索引中删除；
    remove_library=false 时只删除 Milvus 向量。"""
    if remove_library:
        return _dump(remove_paper(paper_id))
    return _dump(delete_paper_vectors(paper_id))


@mcp.tool()
def rag_index_missing() -> str:
    """增量更新：把论文库中尚未建立索引（或索引过期）的论文全部写入 Milvus。"""
    return _dump(index_missing())


@mcp.tool()
def rag_list_papers() -> str:
    """列出论文库中的论文及其索引状态。"""
    indexed = set(get_store().paper_ids()) if get_store().available else set()
    papers = [
        {
            "paper_id": p["id"],
            "title": p["title"][:60],
            "arxiv_id": p.get("arxiv_id"),
            "classification": p.get("classification"),
            "status": p.get("status"),
            "indexed": p["id"] in indexed,
        }
        for p in list_papers()
    ]
    return _dump({"ok": True, "count": len(papers), "papers": papers})


@mcp.tool()
def rag_status() -> str:
    """RAG 系统状态：Milvus 连接、集合向量数、已索引论文数、embedding 模型状态。"""
    return _dump(status())


def main() -> None:
    mcp_cfg = load_rag_config().mcp
    if mcp_cfg.transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run()  # stdio


if __name__ == "__main__":
    main()

"""检索问答（工作流检索注入 + agent loop 追问检索）：

retrieve 节点：混合检索 top_k=6 → 去重 ≤3 篇 → 拼装带编号片段；
answer 节点：prebuilt agent（持有 search_library 工具，可追问检索）按 QA_SYSTEM 作答并标注引用；
结束落 qa_log。
"""

from __future__ import annotations

import json
import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..db import log_qa
from ..llm import get_llm
from ..prompts import QA_SYSTEM, QA_USER_TEMPLATE
from ..retrieval import search
from ..tools import search_library
from ._compat import get_create_agent

logger = logging.getLogger("paper_agent")

MAX_PAPERS = 3  # 引用来源最多去重到 3 篇
SNIPPET_LIMIT = 500  # 每片段字符数
MAX_FOLLOWUP_ROUNDS = 2  # agent 追问检索的最大工具轮数（由 agent 自身决定，这里仅限流）


class QAState(TypedDict, total=False):
    question: str
    context: str
    hits: list[dict]
    answer: str
    citations: list[dict]
    error: str


def retrieve_node(state: QAState) -> dict:
    try:
        hits = search(state["question"], top_k=6)
        if not hits:
            return {"hits": [], "context": ""}
        # 去重到 ≤3 篇论文，保持得分序
        seen_papers: set[int] = set()
        selected = []
        for h in hits:
            if h.paper_id in seen_papers:
                continue
            seen_papers.add(h.paper_id)
            selected.append(h)
            if len(selected) >= MAX_PAPERS:
                break
        snippets = []
        for i, h in enumerate(selected, 1):
            heading = f"（{h.heading}）" if h.heading else ""
            snippets.append(f"[{i}] 《{h.title}》{heading}\n{h.snippet[:SNIPPET_LIMIT]}")
        return {
            "hits": [h.to_dict() for h in selected],
            "context": "\n\n".join(snippets),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"检索失败：{e}"}


def _build_answerer():
    create_agent = get_create_agent()
    return create_agent(
        model=get_llm(),
        tools=[search_library],
        system_prompt=QA_SYSTEM,
    )


def answer_node(state: QAState) -> dict:
    try:
        prompt = QA_USER_TEMPLATE.format(
            question=state["question"], context=state["context"] or "（论文库中没有检索到相关内容）"
        )
        answerer = _build_answerer()
        result = answerer.invoke(
            {"messages": [("user", prompt)]},
            config={"recursion_limit": 25},
        )
        answer = ""
        for msg in reversed(result.get("messages", [])):
            if msg.type == "ai" and msg.content:
                answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        # 引用收集：agent 追问检索的命中优先，否则用初始检索命中
        citations = _collect_citations(result.get("messages", [])) or state.get("hits", [])

        log_qa(state["question"], answer, citations)
        return {"answer": answer, "citations": citations}
    except Exception as e:  # noqa: BLE001
        return {"error": f"回答生成失败：{e}"}


def _collect_citations(messages: list) -> list[dict]:
    """从 agent 的 search_library 工具结果中收集引用来源。"""
    citations: list[dict] = []
    seen: set[tuple] = set()
    for msg in messages:
        if msg.type != "tool" or msg.name != "search_library":
            continue
        content = msg.content
        if not isinstance(content, str):
            continue
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            continue
        for hit in data.get("hits", []):
            key = (hit.get("paper_id"), hit.get("chunk_seq"))
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "paper_id": hit.get("paper_id"),
                    "arxiv_id": hit.get("arxiv_id", ""),
                    "title": hit.get("title", ""),
                    "chunk_seq": hit.get("chunk_seq"),
                }
            )
    return citations


# ---------- 图组装 ----------

def build_qa_graph(checkpointer=None):
    g = StateGraph(QAState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("answer", answer_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile(checkpointer=checkpointer)


def run_qa(question: str, checkpointer=None) -> dict:
    """便捷入口：检索 + 回答，返回最终状态。"""
    graph = build_qa_graph(checkpointer)
    return graph.invoke({"question": question})

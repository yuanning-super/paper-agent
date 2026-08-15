"""解读报告（agent loop 探索 + 工作流收尾）：

阶段一 prebuilt agent loop：理解用户请求 → 入库 / 找 GitHub / 分析仓库，产出 paper_id；
阶段二 工作流节点：确定论文 → 组装全文与仓库素材 → 按 8 章节模板生成报告 → 落盘。
"""

from __future__ import annotations

import json
import logging
import re
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from ..config import load_settings
from ..db import get_chunks, get_paper, update_paper
from ..llm import get_llm
from ..prompts import INTERPRET_AGENT_SYSTEM, REPORT_SYSTEM, REPORT_USER_TEMPLATE
from ..tools import (
    analyze_github_repo,
    arxiv_fetch_metadata,
    find_github_url,
    get_paper_summary,
    ingest_arxiv_paper,
    search_library,
)
from ..utils import truncate
from ._compat import get_create_agent

logger = logging.getLogger("paper_agent")

PAPER_ID_RE = re.compile(r"#\s*(\d+)|论文\s*ID[：:\s]*(\d+)|paper_id[：:\s]*(\d+)")
ARXIV_ID_RE = re.compile(r"(?:arxiv:)?(\d{4}\.\d{4,5})", re.I)

REPORT_SECTIONS = [
    "## 1. 背景与动机",
    "## 2. 核心贡献",
    "## 3. 方法与技术细节",
    "## 4. 实验结果",
    "## 5. 局限与不足",
    "## 6. 相关工作对比",
    "## 7. 代码仓库分析",
    "## 8. 一句话总结",
]


class InterpretState(TypedDict, total=False):
    query: str
    paper_id: int
    error: str
    agent_output: str
    github_material: str
    report_text: str
    report_path: str


# ---------- 阶段一：探索 agent ----------

def _build_explorer(checkpointer=None):
    create_agent = get_create_agent()
    return create_agent(
        model=get_llm(),
        tools=[
            search_library,
            arxiv_fetch_metadata,
            ingest_arxiv_paper,
            find_github_url,
            analyze_github_repo,
            get_paper_summary,
        ],
        system_prompt=INTERPRET_AGENT_SYSTEM,
        checkpointer=checkpointer,
    )


def _extract_paper_id(text: str) -> int | None:
    m = PAPER_ID_RE.search(text)
    if m:
        return int(next(g for g in m.groups() if g))
    return None


def _scan_tool_messages(state: dict) -> int | None:
    """从 agent 的工具消息 JSON 中挖掘 paper_id（模型可能未在最终文本中复述）。"""
    for msg in reversed(state.get("messages", [])):
        if msg.type == "tool":
            content = msg.content
            if isinstance(content, str):
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and data.get("paper_id"):
                        return int(data["paper_id"])
                except (json.JSONDecodeError, ValueError):
                    pass
    return None


def prepare_node(state: InterpretState) -> dict:
    try:
        query = state["query"]
        # 确定性短路：查询已指明库内论文 ID 或 arXiv ID（已在库中）时，跳过探索 agent
        m_paper = re.search(r"paper[:#\s]*(\d+)", query)
        if m_paper:
            paper_id = int(m_paper.group(1))
            if get_paper(paper_id):
                return {"paper_id": paper_id, "agent_output": f"已定位库内论文 #{paper_id}"}
        m_arxiv = ARXIV_ID_RE.search(query)
        if m_arxiv:
            from ..db import find_paper_by_arxiv_id

            paper = find_paper_by_arxiv_id(m_arxiv.group(1))
            if paper:
                return {"paper_id": paper["id"], "agent_output": f"已定位库内论文 #{paper['id']}"}

        explorer = _build_explorer()
        result = explorer.invoke(
            {"messages": [("user", state["query"])]},
            config={"recursion_limit": 30},
        )
        messages = result.get("messages", [])
        final_text = ""
        for msg in reversed(messages):
            if msg.type == "ai" and msg.content:
                final_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        paper_id = _extract_paper_id(final_text)
        if paper_id is None:
            paper_id = _scan_tool_messages(result)

        if paper_id is None:
            # 兜底：查询本身含 arXiv ID
            m = ARXIV_ID_RE.search(state["query"])
            if m:
                from ..db import find_paper_by_arxiv_id

                paper = find_paper_by_arxiv_id(m.group(1))
                if paper:
                    paper_id = paper["id"]
        if paper_id is None:
            return {"error": "无法确定要解读的论文。请提供 arXiv ID 或论文库中的论文 ID。", "agent_output": final_text}
        return {"paper_id": paper_id, "agent_output": final_text}
    except Exception as e:  # noqa: BLE001
        return {"error": f"论文资料准备失败：{e}"}


# ---------- 阶段二：报告生成 ----------

def report_node(state: InterpretState) -> dict:
    try:
        settings = load_settings()
        paper = get_paper(state["paper_id"])
        if not paper:
            return {"error": f"论文 #{state['paper_id']} 不存在"}

        # 全文组装（60k 字符预算，头尾保留）
        chunks = get_chunks(state["paper_id"])
        full_text = "\n\n".join(c["content"] for c in chunks)
        full_text = truncate(full_text, settings.full_text_budget, mode="head_tail")

        # GitHub 素材：未记录链接时先查找，找到后分析仓库
        github_material = ""
        try:
            if not paper.get("github_url"):
                found = find_github_url.invoke({"paper_id": state["paper_id"]})
                data = json.loads(str(found))
                if data.get("url"):
                    paper = get_paper(state["paper_id"])  # 重新读取回填后的记录
            if paper.get("github_url"):
                analyzed = analyze_github_repo.invoke({"url": paper["github_url"]})
                github_material = str(analyzed)
        except Exception as e:  # noqa: BLE001
            logger.warning("GitHub 查找/分析失败：%s", e)
            github_material = f"GitHub 查找/分析失败：{e}"

        prompt = REPORT_USER_TEMPLATE.format(
            title=paper.get("title", ""),
            authors=", ".join(paper.get("authors", [])[:10]),
            arxiv_id=paper.get("arxiv_id") or "—",
            published=paper.get("published") or "—",
            categories=", ".join(paper.get("categories", [])),
            abstract=(paper.get("abstract") or "")[:1500],
            full_text=full_text,
            github_material=github_material or "（未找到官方代码仓库）",
        )

        # 用 invoke（而非手动 stream）：LangGraph 的 stream_mode="messages" 才能拿到 token 流
        resp = get_llm(
            max_tokens=settings.report_max_tokens,
            temperature=settings.report_temperature,
        ).invoke([SystemMessage(content=REPORT_SYSTEM), HumanMessage(content=prompt)])
        report_text = resp.content if isinstance(resp.content, str) else str(resp.content)

        # 章节完整性兜底：缺哪节补哪节标题（模型偶发省略标题时不丢结构）
        for section in REPORT_SECTIONS:
            alt = section.replace("## 1. ", "## ").split(". ", 1)[-1]
            if section not in report_text and alt not in report_text:
                report_text += f"\n\n{section}\n（本节内容缺失）\n"

        report_path = settings.reports_dir / f"{state['paper_id']}.md"
        report_path.write_text(
            f"# 论文解读：{paper.get('title', '')}\n\n{report_text}\n", encoding="utf-8"
        )
        update_paper(
            state["paper_id"],
            report_path=str(report_path),
            status="interpreted",
        )
        return {"report_text": report_text, "report_path": str(report_path)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"报告生成失败：{e}"}


def _router(state: InterpretState) -> str:
    return END if state.get("error") else "report"


# ---------- 图组装 ----------

def build_interpret_graph(checkpointer=None):
    g = StateGraph(InterpretState)
    g.add_node("prepare", prepare_node)
    g.add_node("report", report_node)
    g.add_edge(START, "prepare")
    g.add_conditional_edges("prepare", _router, {"report": "report", END: END})
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)

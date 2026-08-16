"""解读流水线：load（定位论文 + 章节拆分 + 仓库素材）→ 强制委派四个子 agent。

- background：实验背景调研（数据集与 baseline）
- method：方法创新性与实现分析（代码辅助）
- experiment：实验结果与分析（结合背景调研）
- report：综合三份子报告与原文，输出最终解读报告
"""

from __future__ import annotations

import json
import logging
import operator
import re
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph

from ..config import load_settings
from ..db import find_paper_by_arxiv_id, get_paper, update_paper
from ..ingestion import extract_figures, extract_text
from ..llm import get_llm
from ..prompts import (
    BACKGROUND_AGENT_SYSTEM,
    BACKGROUND_USER_TEMPLATE,
    EXPERIMENT_AGENT_SYSTEM,
    EXPERIMENT_USER_TEMPLATE,
    METHOD_AGENT_SYSTEM,
    METHOD_USER_TEMPLATE,
    REPORT_AGENT_SYSTEM,
    REPORT_USER_TEMPLATE,
)
from ..rag.chunking import split_sections
from ..tools import (
    analyze_github_repo,
    arxiv_fetch_metadata,
    find_github_url,
    get_paper_summary,
    ingest_arxiv_paper,
    read_repo_file,
    read_section,
    search_library,
    set_section_cache,
)
from ..utils import truncate
from ._compat import get_create_agent

logger = logging.getLogger("paper_agent")

PAPER_ID_RE = re.compile(r"#\s*(\d+)|论文\s*ID[：:\s]*(\d+)|paper_id[：:\s]*(\d+)")
ARXIV_ID_RE = re.compile(r"(?:arxiv:)?(\d{4}\.\d{4,5})", re.IGNORECASE)

# 论文定位 agent（自然语言查询时用）
RESOLVE_SYSTEM = (
    "你在一个论文库中。根据用户描述定位要解读的论文：可用 search_library 检索、"
    "get_paper_summary 确认、arxiv_fetch_metadata 查元数据；论文未入库时先 ingest_arxiv_paper 入库。"
    "最后只回复一行：论文 ID #N（N 为数字）。"
)


class InterpretState(TypedDict, total=False):
    query: str
    paper_id: int
    error: str
    events: Annotated[list[str], operator.add]
    paper_map: str
    repo_material: str
    background: str
    method_analysis: str
    experiment_analysis: str
    report_text: str
    report_path: str


# ---------- 工具函数 ----------

def _run_subagent(system: str, user: str, tools: list) -> str:
    """运行一个 prebuilt 子 agent，返回其最终文本。

    config 继承父级（含回调管理器），否则外层 stream_mode="messages" 拿不到子 agent 的 token 流。
    """
    create_agent = get_create_agent()
    agent = create_agent(
        model=get_llm(max_tokens=6000),
        tools=tools,
        system_prompt=system,
    )
    result = agent.invoke(
        {"messages": [("user", user)]},
        config={**get_config(), "recursion_limit": 30},
    )
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _join_role(sections: list[dict], role: str, limit: int) -> str:
    """把指定角色的章节内容拼接并截断（角色由 split_sections 计算，含父级继承）。"""
    parts = [f"### {s['title']}\n{s['content']}" for s in sections if s.get("role") == role]
    return truncate("\n\n".join(parts), limit, mode="head") if parts else "（无）"


def _resolve_paper_id(query: str) -> tuple[int | None, str]:
    """定位论文：短路匹配 → 定位 agent → arXiv 兜底。"""
    m = re.search(r"paper[:#\s]*(\d+)", query)
    if m and get_paper(int(m.group(1))):
        return int(m.group(1)), f"已定位库内论文 #{m.group(1)}"
    m = ARXIV_ID_RE.search(query)
    if m:
        paper = find_paper_by_arxiv_id(m.group(1))
        if paper:
            return paper["id"], f"已定位库内论文 #{paper['id']}"

    agent = get_create_agent()(
        model=get_llm(),
        tools=[search_library, get_paper_summary, arxiv_fetch_metadata, ingest_arxiv_paper],
        system_prompt=RESOLVE_SYSTEM,
    )
    result = agent.invoke(
        {"messages": [("user", query)]},
        config={**get_config(), "recursion_limit": 30},
    )
    final_text = ""
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content:
            final_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    mm = PAPER_ID_RE.search(final_text)
    if mm:
        return int(next(g for g in mm.groups() if g)), final_text
    return None, final_text


# ---------- 节点 ----------

def load_node(state: InterpretState) -> dict:
    try:
        paper_id, note = _resolve_paper_id(state["query"])
        if paper_id is None:
            return {"error": "无法确定要解读的论文。请提供 arXiv ID 或论文库中的论文 ID。"}
        paper = get_paper(paper_id)
        events = [note]

        # 章节拆分：从 PDF 提取并切分，写入缓存供 read_section 按需读取
        sections = []
        fig_count = 0
        if paper.get("pdf_path") and Path(paper["pdf_path"]).exists():
            text, _ = extract_text(Path(paper["pdf_path"]))
            sections = split_sections(text)
            figures = extract_figures(Path(paper["pdf_path"]), paper_id)
            fig_count = len(figures)
        if not sections:
            sections = [{"title": "全文", "content": (paper.get("abstract") or ""), "role": "other"}]
        set_section_cache(paper_id, sections)
        events.append(f"章节拆分完成：{len(sections)} 节")
        if fig_count:
            events.append(f"原文图表已提取：{fig_count} 张")

        # 论文地图：摘要 + 章节目录（编号与 read_section 一致）
        paper_map = "\n".join(
            f"{i}. {s['title']}（{s.get('role', 'other')}，{len(s['content'])} 字）"
            for i, s in enumerate(sections)
        )

        # GitHub 仓库：查找链接并分析
        repo_material = ""
        if not paper.get("github_url"):
            try:
                found = json.loads(str(find_github_url.invoke({"paper_id": paper_id})))
                if found.get("url"):
                    paper = get_paper(paper_id)  # 重新读取回填后的记录
            except (json.JSONDecodeError, ValueError):
                pass
        if paper.get("github_url"):
            try:
                repo_material = str(analyze_github_repo.invoke({"url": paper["github_url"]}))
                events.append(f"代码仓库已分析：{paper['github_url']}")
            except Exception as e:  # noqa: BLE001
                logger.warning("仓库分析失败：%s", e)

        return {
            "paper_id": paper_id,
            "paper_map": paper_map,
            "repo_material": repo_material,
            "events": events,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"论文资料准备失败：{e}"}


def background_node(state: InterpretState) -> dict:
    if state.get("error"):
        return {}
    try:
        paper = get_paper(state["paper_id"])
        sections = _sections(state["paper_id"])
        user = BACKGROUND_USER_TEMPLATE.format(
            title=paper["title"],
            abstract=truncate(paper.get("abstract") or "", 2000, mode="head"),
            experiment=_join_role(sections, "experiment", 8000),
            related=_join_role(sections, "related", 4000),
            references=_join_role(sections, "references", 6000),
        )
        out = _run_subagent(
            BACKGROUND_AGENT_SYSTEM, user, [read_section, search_library, get_paper_summary]
        )
        return {"background": out, "events": ["✅ 背景调研完成（数据集与 baseline）"]}
    except Exception as e:  # noqa: BLE001 —— 并行节点失败不中断整体，写入输出字段由报告 agent 如实说明
        return {"background": f"（背景调研失败：{e}）", "events": [f"背景调研失败：{e}"]}


def method_node(state: InterpretState) -> dict:
    if state.get("error"):
        return {}
    try:
        paper = get_paper(state["paper_id"])
        sections = _sections(state["paper_id"])
        user = METHOD_USER_TEMPLATE.format(
            title=paper["title"],
            intro=_join_role(sections, "intro", 3000),
            method=_join_role(sections, "method", 10_000),
            repo_material=truncate(state.get("repo_material", ""), 8000, mode="head") or "（无）",
        )
        out = _run_subagent(
            METHOD_AGENT_SYSTEM, user, [read_section, read_repo_file, search_library]
        )
        return {"method_analysis": out, "events": ["✅ 方法分析完成（创新性与实现）"]}
    except Exception as e:  # noqa: BLE001
        return {"method_analysis": f"（方法分析失败：{e}）", "events": [f"方法分析失败：{e}"]}


def experiment_node(state: InterpretState) -> dict:
    if state.get("error"):
        return {}
    try:
        paper = get_paper(state["paper_id"])
        sections = _sections(state["paper_id"])
        user = EXPERIMENT_USER_TEMPLATE.format(
            title=paper["title"],
            experiment=_join_role(sections, "experiment", 8000),
        )
        out = _run_subagent(EXPERIMENT_AGENT_SYSTEM, user, [read_section])
        return {"experiment_analysis": out, "events": ["✅ 实验分析完成"]}
    except Exception as e:  # noqa: BLE001
        return {"experiment_analysis": f"（实验分析失败：{e}）", "events": [f"实验分析失败：{e}"]}


def report_node(state: InterpretState) -> dict:
    if state.get("error"):
        return {"error": state["error"]}
    try:
        settings = load_settings()
        paper = get_paper(state["paper_id"])
        user = REPORT_USER_TEMPLATE.format(
            title=paper["title"],
            authors=", ".join(paper.get("authors", [])[:10]),
            arxiv_id=paper.get("arxiv_id") or "—",
            published=paper.get("published") or "—",
            abstract=(paper.get("abstract") or "")[:1500],
            sections_map=state.get("paper_map", ""),
            background=truncate(state.get("background", ""), 10_000, mode="head"),
            method_analysis=truncate(state.get("method_analysis", ""), 10_000, mode="head"),
            experiment_analysis=truncate(state.get("experiment_analysis", ""), 10_000, mode="head"),
            repo_material=truncate(state.get("repo_material", ""), 8000, mode="head") or "（未找到官方代码仓库）",
        )
        out = _run_subagent(REPORT_AGENT_SYSTEM, user, [read_section])

        # 保存各阶段结果（与最终报告一并落盘，供回溯查阅）
        step_names = {
            "background": ("背景调研", state.get("background", "")),
            "method": ("方法分析", state.get("method_analysis", "")),
            "experiment": ("实验分析", state.get("experiment_analysis", "")),
        }
        for key, (label, content) in step_names.items():
            (settings.reports_dir / f"{state['paper_id']}.{key}.md").write_text(
                f"# {label}\n\n{content}\n", encoding="utf-8"
            )

        report_path = settings.reports_dir / f"{state['paper_id']}.md"
        report_path.write_text(
            f"# 论文解读：{paper.get('title', '')}\n\n{out}\n", encoding="utf-8"
        )
        update_paper(state["paper_id"], report_path=str(report_path), status="interpreted")
        return {
            "report_text": out,
            "report_path": str(report_path),
            "events": ["✅ 解读报告已生成（含各阶段结果）"],
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"报告生成失败：{e}"}


def _sections(paper_id: int) -> list[dict]:
    from ..tools import get_section_cache

    return get_section_cache(paper_id)


# ---------- 图组装 ----------

def build_interpret_graph(checkpointer=None):
    g = StateGraph(InterpretState)
    g.add_node("load", load_node)
    g.add_node("background", background_node)
    g.add_node("method", method_node)
    g.add_node("experiment", experiment_node)
    g.add_node("report", report_node)
    g.add_edge(START, "load")
    # 三个分析子 agent 互不依赖 → 并行执行（fan-out）；report 等待三路完成（fan-in）
    g.add_edge("load", "background")
    g.add_edge("load", "method")
    g.add_edge("load", "experiment")
    g.add_edge("background", "report")
    g.add_edge("method", "report")
    g.add_edge("experiment", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)

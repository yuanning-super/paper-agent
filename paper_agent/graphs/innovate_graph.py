"""创新工坊工作流：创新点抽取（校验重试）与跨论文想法生成（硬校验引用）可分可合。

- build_extract_graph：extract → END（只抽取创新点入库）
- build_ideas_graph：generate → END（用库中已有创新点生成想法）
- build_innovate_graph：extract → generate → END（先抽取再生成）
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..db import (
    get_chunks,
    get_innovations,
    get_paper,
    save_ideas,
    save_innovations,
)
from ..llm import complete_json
from ..prompts import IDEAS_GENERATE_PROMPT, INNOVATION_EXTRACT_PROMPT

logger = logging.getLogger("paper_agent")

VALID_KINDS = {"method", "motivation", "finding", "design", "dataset"}


class InnovateState(TypedDict, total=False):
    paper_ids: list[int]
    innovations: list[dict]  # 已抽取并入库的创新点（跨论文）
    ideas: list[dict]
    error: str
    events: list[str]


# ---------- 校验器 ----------

def _validate_innovation_list(data: Any, valid_chunk_ids: set[int]) -> str | None:
    if not isinstance(data, list):
        return "输出必须是数组"
    if not 1 <= len(data) <= 10:
        return f"创新点数量必须在 1-10 条，实际 {len(data)} 条"
    for it in data:
        if not isinstance(it, dict):
            return "每条创新点必须是对象"
        if it.get("kind") not in VALID_KINDS:
            return f"kind 非法：{it.get('kind')}（须为 {'/'.join(sorted(VALID_KINDS))}）"
        if not it.get("title") or len(str(it["title"])) > 20:
            return f"title 缺失或超过 20 字：{it.get('title')}"
        if not it.get("description"):
            return "description 缺失"
        if not it.get("novelty"):
            return "novelty 缺失"
        ids = it.get("source_chunk_ids")
        if not isinstance(ids, list) or not ids:
            return f"source_chunk_ids 缺失：{it.get('title')}"
        bad = [i for i in ids if i not in valid_chunk_ids]
        if bad:
            return f"引用了不存在的分块序号 {bad}（合法序号集合大小 {len(valid_chunk_ids)}）"
    return None


def _validate_idea_list(data: Any, valid_innovation_ids: set[int]) -> str | None:
    if not isinstance(data, list):
        return "输出必须是数组"
    if not 3 <= len(data) <= 5:
        return f"想法数量必须在 3-5 个，实际 {len(data)} 个"
    for it in data:
        if not isinstance(it, dict):
            return "每个想法必须是对象"
        if not it.get("title") or not it.get("hypothesis") or not it.get("combination"):
            return f"title/hypothesis/combination 缺失：{it.get('title')}"
        ids = it.get("source_innovation_ids")
        if not isinstance(ids, list) or len(ids) < 2:
            return f"source_innovation_ids 必须 ≥2 个：{it.get('title')}"
        bad = [i for i in ids if i not in valid_innovation_ids]
        if bad:
            return f"引用了不在输入集合中的创新点 ID {bad}（合法：{sorted(valid_innovation_ids)}）"
        if not it.get("feasibility"):
            return "feasibility 缺失"
        if not isinstance(it.get("risks"), list) or len(it.get("risks")) < 2:
            return "risks 必须 ≥2 条"
        if not isinstance(it.get("experiments"), list) or not it.get("experiments"):
            return "experiments 缺失"
    return None


# ---------- 节点 ----------

def extract_node(state: InnovateState) -> dict:
    """对每篇论文抽取创新点并入库。"""
    paper_ids = state["paper_ids"]
    if not paper_ids:
        return {"error": "未指定论文"}
    events: list[str] = []
    all_innovations: list[dict] = []
    for paper_id in paper_ids:
        try:
            paper = get_paper(paper_id)
            if not paper:
                events.append(f"论文 #{paper_id} 不存在，已跳过")
                continue
            chunks = get_chunks(paper_id)
            if not chunks:
                events.append(f"《{paper['title'][:40]}》尚未分块，请先入库")
                continue
            valid_ids = {c["seq"] for c in chunks}
            chunk_text = "\n\n".join(f"[{c['seq']}] {c['content']}" for c in chunks)
            prompt = INNOVATION_EXTRACT_PROMPT.format(
                title=paper["title"], chunks=chunk_text[:50_000]
            )
            data, _ = complete_json(
                prompt,
                validator=lambda d: _validate_innovation_list(d, valid_ids),
            )
            if data is None:
                events.append(f"《{paper['title'][:40]}》创新点抽取失败（重试后仍无法解析）")
                continue
            save_innovations(paper_id, data)
            for it in data:
                it = dict(it)
                it["paper_id"] = paper_id
                all_innovations.append(it)
            events.append(f"《{paper['title'][:40]}》抽取 {len(data)} 条创新点")
        except Exception as e:  # noqa: BLE001 —— 单篇失败不中断整体
            events.append(f"论文 #{paper_id} 抽取异常：{e}")
    if not all_innovations:
        return {"error": "没有成功抽取任何创新点", "events": events}
    return {"innovations": all_innovations, "events": events}


def generate_node(state: InnovateState) -> dict:
    """跨论文创新点组合 → 新研究想法（引用硬校验）。"""
    try:
        innovations = state["innovations"]
        # 重新从库中取一遍，保证每条带数据库 id（抽取节点入库后才有 id）
        saved: list[dict] = []
        for paper_id in sorted({it["paper_id"] for it in innovations}):
            saved.extend(get_innovations(paper_id))
        if saved:
            innovations = saved
        valid_ids = {it["id"] for it in innovations if it.get("id")}
        if len(valid_ids) < 2:
            return {"error": "可用创新点不足 2 条，无法生成研究想法"}

        listing = "\n".join(
            f"- id={it['id']} | 论文《{_paper_title(it['paper_id'])}》 | kind={it['kind']} | "
            f"标题：{it['title']}\n  描述：{it['description']}"
            for it in innovations
        )
        prompt = IDEAS_GENERATE_PROMPT.format(innovations=listing[:20_000])
        data, _ = complete_json(
            prompt,
            validator=lambda d: _validate_idea_list(d, valid_ids),
            max_tokens=6000,
        )
        if data is None:
            return {"error": "研究想法生成失败（重试后仍无法解析或校验不通过）"}

        # 补全 source_paper_ids（从创新点反查）
        innov_paper = {it["id"]: it["paper_id"] for it in innovations if it.get("id")}
        for idea in data:
            paper_ids = sorted(
                {innov_paper[i] for i in idea["source_innovation_ids"] if i in innov_paper}
            )
            idea["source_paper_ids"] = idea.get("source_paper_ids") or paper_ids

        idea_ids = save_ideas(data)
        for idea, iid in zip(data, idea_ids):
            idea["id"] = iid
        return {
            "ideas": data,
            "events": [f"生成 {len(data)} 个新研究想法（ID：{', '.join(map(str, idea_ids))}）"],
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"研究想法生成失败：{e}"}


def _paper_title(paper_id: int) -> str:
    paper = get_paper(paper_id)
    return (paper["title"][:30] + "…") if paper else f"#{paper_id}"


# ---------- 图组装 ----------

def _make_router(next_node: str):
    def router(state: InnovateState) -> str:
        return END if state.get("error") else next_node

    return router


def build_extract_graph(checkpointer=None):
    g = StateGraph(InnovateState)
    g.add_node("extract", extract_node)
    g.add_edge(START, "extract")
    g.add_edge("extract", END)
    return g.compile(checkpointer=checkpointer)


def build_ideas_graph(checkpointer=None):
    g = StateGraph(InnovateState)
    g.add_node("generate", generate_node)
    g.add_edge(START, "generate")
    g.add_edge("generate", END)
    return g.compile(checkpointer=checkpointer)


def build_innovate_graph(checkpointer=None):
    g = StateGraph(InnovateState)
    g.add_node("extract", extract_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "extract")
    g.add_conditional_edges("extract", _make_router("generate"), {"generate": "generate", END: END})
    g.add_edge("generate", END)
    return g.compile(checkpointer=checkpointer)


# ---------- 便捷入口 ----------

def run_extract(paper_ids: list[int], checkpointer=None) -> dict:
    """只抽取创新点并入库。"""
    return build_extract_graph(checkpointer).invoke({"paper_ids": paper_ids})


def run_ideas(paper_ids: list[int], checkpointer=None) -> dict:
    """用论文库中已有的创新点生成研究想法（不重新抽取）。"""
    innovations: list[dict] = []
    for paper_id in paper_ids:
        innovations.extend(get_innovations(paper_id))
    return build_ideas_graph(checkpointer).invoke(
        {"paper_ids": paper_ids, "innovations": innovations}
    )


def run_innovate(paper_ids: list[int], checkpointer=None) -> dict:
    """先抽取创新点，再生成研究想法。"""
    return build_innovate_graph(checkpointer).invoke({"paper_ids": paper_ids})

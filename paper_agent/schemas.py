"""Pydantic 数据模型：创新点、研究想法、元数据增强。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

InnovationKind = Literal["method", "motivation", "finding", "design", "dataset"]

CLASSIFICATION_OPTIONS = [
    "深度学习",
    "自然语言处理",
    "计算机视觉",
    "语音",
    "强化学习",
    "优化算法",
    "系统与分布式",
    "机器学习理论",
    "其他",
]


class Innovation(BaseModel):
    """单条创新点（抽取自论文）。"""

    kind: InnovationKind = Field(description="创新点类型")
    title: str = Field(description="创新点标题，不超过 20 字")
    description: str = Field(description="创新点描述，60-120 字")
    novelty: str = Field(description="相对已有工作的新意所在")
    source_chunk_ids: list[int] = Field(description="支撑该创新点的原文分块序号列表")

    def validate_chunk_ids(self, valid_ids: set[int]) -> str | None:
        if not self.source_chunk_ids:
            return f"创新点「{self.title}」缺少 source_chunk_ids"
        bad = [i for i in self.source_chunk_ids if i not in valid_ids]
        if bad:
            return f"创新点「{self.title}」引用了不存在的分块序号 {bad}"
        return None


class ResearchIdea(BaseModel):
    """由多个创新点组合/拓展生成的新研究想法。"""

    title: str = Field(description="研究想法标题")
    hypothesis: str = Field(description="核心假设/核心思想")
    combination: str = Field(description="如何组合来源创新点")
    source_innovation_ids: list[int] = Field(description="引用的创新点 ID 列表（至少 2 个）")
    source_paper_ids: list[int] = Field(description="来源论文 ID 列表")
    feasibility: str = Field(description="可行性评估：高/中/低 + 理由")
    risks: list[str] = Field(description="潜在风险，至少 2 条")
    experiments: list[str] = Field(description="可执行的实验设计建议")

    def validate_sources(self, valid_innovation_ids: set[int]) -> str | None:
        if len(self.source_innovation_ids) < 2:
            return f"想法「{self.title}」只引用了 {len(self.source_innovation_ids)} 个创新点，必须 ≥2 个"
        bad = [i for i in self.source_innovation_ids if i not in valid_innovation_ids]
        if bad:
            return f"想法「{self.title}」引用了不在输入集合中的创新点 ID {bad}"
        if len(self.risks) < 2:
            return f"想法「{self.title}」的风险列表少于 2 条"
        return None


class Enrichment(BaseModel):
    """LLM 元数据增强输出。"""

    title_zh: str = Field(description="论文标题的中文翻译")
    keywords: list[str] = Field(description="5-8 个中文关键词")
    classification: str = Field(
        description="研究方向分类，候选：" + " / ".join(CLASSIFICATION_OPTIONS)
    )

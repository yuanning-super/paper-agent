"""文档分块与章节拆分：字符窗口分块（检索用）+ 章节切分（解读用）。"""

from __future__ import annotations

import re

from ..db import Chunk
from .config import load_rag_config


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    cfg = load_rag_config().chunking
    size = size or cfg.size
    overlap = overlap or cfg.overlap
    chunks: list[Chunk] = []
    current_heading: str | None = None
    for i, piece in enumerate(_split(text, size, overlap)):
        heading = _detect_heading(piece.splitlines()[0])
        if heading:
            current_heading = heading
        chunks.append(Chunk(paper_id=0, seq=i, content=piece, heading=current_heading))
    return chunks


def _split(text: str, size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:  # 尽量在句末/换行处断开
            window = text[start:end]
            for sep in ("。\n", ".\n", "。", ". ", "；", "\n\n"):
                pos = window.rfind(sep)
                if pos > size // 2:
                    end = start + pos + len(sep)
                    break
        pieces.append(text[start:end].strip())
        if end >= n:
            break
        start = end - overlap
    return [p for p in pieces if p]


def _detect_heading(line: str) -> str | None:
    line = line.strip()
    if not line or len(line) > 60:
        return None
    if re.match(r"^\s*(?:第[一二三四五六七八九十\d]+[章节部分]|[0-9]+(?:\.[0-9]+)*)\s*\S", line):
        return line
    if re.match(
        r"^\s*(?:Abstract|Introduction|Related Work|Method|Experiments?|Results?|Conclusion|References)\b",
        line,
        re.IGNORECASE,
    ):
        return line
    return None


# ---------- 章节拆分（解读流水线用） ----------

# 编号式标题：1 Introduction / 3.1. Model / II. Method / 第一章
# 注意：不能加 IGNORECASE，否则 [A-Z] 会匹配小写开头（"6.9 days…" 类表格行）
_SECTION_HEADING_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\.?\s+[A-Z一-鿿][^\n]{1,59}"
    r"|[IVX]+\.\s+[A-Z][^\n]{1,59}"
    r"|第[一二三四五六七八九十\d]+[章节部分]"
    r")",
)

# 关键词式标题：整行短标题（"Abstract"、"Model Architecture" 等裸标题行，
# 很多论文的编号在版面边距中不会进入提取文本）
_KEYWORD_HEADING_RE = re.compile(
    r"^(?:"
    r"Abstract|Introduction|Related\s+Work|Background|Preliminaries?|Methodology|Methods?|"
    r"Approach|Model|Architecture|Algorithm|Framework|Training|Implementation|"
    r"Experimental\s+Setup|Experiments?|Evaluation|Results?|Ablation|Discussion|Conclusion|"
    r"Future\s+Work|References|Bibliography|Appendix|Acknowledgments?"
    r"|摘要|引言|绪论|相关工作|背景|预备知识|方法|模型|架构|算法|框架|训练|实现细节|"
    r"实验设置|实验|评估|结果|消融|讨论|结论|展望|参考文献|附录|致谢"
    r")(?:\s+\S+){0,4}[:：]?$",
    re.IGNORECASE,
)

# 单单词标题白名单：排除 "Model"/"Training" 等常作表格列头的单词
_BARE_TITLES = {
    "abstract", "introduction", "background", "preliminary", "methodology", "method", "methods",
    "approach", "experiments", "evaluation", "results", "ablation", "discussion", "conclusion",
    "references", "bibliography", "appendix", "acknowledgments",
    "摘要", "引言", "绪论", "背景", "预备知识", "方法", "模型", "架构", "算法", "框架", "训练",
    "实现细节", "实验设置", "实验", "评估", "结果", "消融", "讨论", "结论", "展望", "参考文献", "附录", "致谢",
}


def _is_section_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
    if _SECTION_HEADING_RE.match(s):
        return True
    # 短行 + 无括号/句末标点（排除表格头、图注与句首关键词的正文句子）
    if len(s) > 40 or "(" in s or ")" in s or s.endswith((".", "!", "?", ",", ";")):
        return False
    if _KEYWORD_HEADING_RE.match(s):
        # 多词标题直接接受；单单词需在白名单内（过滤 "Model" 等表格列头）
        return " " in s or s.lower() in _BARE_TITLES
    # 全大写多词标题（如 "MODEL OVERVIEW"）；排除含数字/括号的表格头噪音
    return s.isupper() and len(s.split()) >= 3 and not any(c.isdigit() for c in s)


def split_sections(text: str) -> list[dict]:
    """按章节标题行切分全文，返回 [{title, content, role}]（保持原文顺序）。

    role 带父级继承：子章节标题无法直接归类时沿用所属大章节的角色
    （如 Transformer 的 "6.3 English Constituency Parsing" 归入 experiment）。
    flush 在父角色更新前调用，故每节记录的是"本节结束时"的父角色——恰为继承语义。
    内容 <200 字的碎片节（表格头、算法框头等噪音）并入下一节，避免碎片章节。
    """
    sections: list[dict] = []
    current_title = ""
    buf: list[str] = []
    parent_role = "other"

    def flush() -> None:
        content = "\n".join(buf).strip()
        if content:
            sections.append({"title": current_title or "正文", "content": content, "role": parent_role})

    for line in text.splitlines():
        if _is_section_heading(line):
            flush()
            current_title = line.strip()
            buf = []
            role = classify_section(current_title)
            if role != "other":
                parent_role = role  # 可识别的新章节刷新父角色；子章节则沿用
        else:
            buf.append(line)
    flush()

    # 碎片节合并：内容过短视为误判（表格头/算法框头），并入后续正常节
    merged: list[dict] = []
    carry = ""
    for sec in sections:
        if len(sec["content"]) < 200:
            carry += f"{sec['title']}\n{sec['content']}\n"
            continue
        if carry:
            sec["content"] = carry + sec["content"]
            carry = ""
        merged.append(sec)
    if carry and merged:
        merged[-1]["content"] += "\n" + carry
    sections = merged

    # 无标题结构（如纯文本 PDF）：退化为按段落切分，保证 read_section 可用
    if len(sections) <= 1:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        sections = [
            {"title": f"段落 {i}", "content": p, "role": "other"}
            for i, p in enumerate(paragraphs)
        ]
    return sections


_ROLE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("abstract", ("abstract", "摘要")),
    ("intro", ("introduction", "引言", "绪论")),
    ("related", ("related work", "background", "preliminar", "相关工作", "背景", "预备")),
    (
        "method",
        ("method", "approach", "model", "architecture", "algorithm", "framework", "training",
         "implementation", "problem", "方法", "模型", "架构", "算法", "框架", "训练", "实现细节"),
    ),
    (
        "experiment",
        ("experiment", "evaluation", "result", "ablation", "setup", "dataset", "benchmark",
         "实验", "评估", "结果", "消融", "设置", "数据集"),
    ),
    ("conclusion", ("conclusion", "discussion", "limitation", "future work", "结论", "讨论", "局限", "展望")),
    ("references", ("reference", "bibliograph", "参考文献")),
]


def classify_section(title: str) -> str:
    """章节标题 → 角色（abstract/intro/related/method/experiment/conclusion/references/other）。"""
    t = title.lower()
    for role, keywords in _ROLE_KEYWORDS:
        if any(k in t for k in keywords):
            return role
    return "other"

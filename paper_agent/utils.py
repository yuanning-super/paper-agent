"""通用工具：JSON 修复解析、文本截断、分块、章节标题启发式。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("paper_agent")


# ---------- JSON 解析 ----------

def extract_json(text: str) -> Any | None:
    """四级修复式 JSON 解析：剥 fence → 整体 loads → 平衡括号切片 → 去尾逗号/控制字符。"""
    if not text:
        return None

    # ① 剥离 markdown 代码块围栏
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    text = text.strip()

    # ② 整体解析
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # ③ 平衡括号切片（首个 { 或 [ 至最后一个对应闭括号）
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        first, last = text.find(open_ch), text.rfind(close_ch)
        if first != -1 and last > first:
            try:
                return json.loads(text[first : last + 1])
            except (json.JSONDecodeError, ValueError):
                pass

    # ④ 去尾逗号、替换控制字符后重试
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        first, last = cleaned.find(open_ch), cleaned.rfind(close_ch)
        if first != -1 and last > first:
            try:
                return json.loads(cleaned[first : last + 1])
            except (json.JSONDecodeError, ValueError):
                pass

    return None


# ---------- 文本处理 ----------

def truncate(text: str, limit: int, mode: str = "head_tail") -> str:
    """截断长文本；head_tail 保留头尾各一半。"""
    if len(text) <= limit:
        return text
    if mode == "head":
        return text[:limit] + "\n…（内容过长已截断）"
    half = limit // 2
    return text[:half] + "\n…（中间内容过长已省略）…\n" + text[-half:]


def split_sentences_zh(text: str, size: int, overlap: int) -> list[str]:
    """按字符窗口分块（中文无空格分词，按字符计数），尽量在句末断开。"""
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        # 尽量在句末/换行处断开
        if end < n:
            window = text[start:end]
            for sep in ("。\n", ".\n", "。", ". ", "；", "\n\n"):
                pos = window.rfind(sep)
                if pos > size // 2:
                    end = start + pos + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = end - overlap
    return [c for c in chunks if c]


def detect_heading(line: str) -> str | None:
    """启发式判断一行是否为章节标题。"""
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

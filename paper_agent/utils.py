"""通用工具：JSON 修复解析、文本截断。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("paper_agent")


# ---------- JSON 解析 ----------

def extract_json(text: str) -> Any | None:
    """修复式 JSON 解析：剥 fence → 整体 loads → 平衡括号切片 → 去尾逗号重试。"""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    for candidate in _candidates(text.strip()):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _candidates(text: str) -> list[str]:
    """按可靠性从高到低产出解析候选。"""
    candidates = [text]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        first, last = text.find(open_ch), text.rfind(close_ch)
        if first != -1 and last > first:
            candidates.append(text[first : last + 1])
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", cleaned).strip()
    candidates.append(cleaned)
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        first, last = cleaned.find(open_ch), cleaned.rfind(close_ch)
        if first != -1 and last > first:
            candidates.append(cleaned[first : last + 1])
    return candidates


# ---------- 文本处理 ----------

def truncate(text: str, limit: int, mode: str = "head_tail") -> str:
    """截断长文本；head_tail 保留头尾各一半。"""
    if len(text) <= limit:
        return text
    if mode == "head":
        return text[:limit] + "\n…（内容过长已截断）"
    half = limit // 2
    return text[:half] + "\n…（中间内容过长已省略）…\n" + text[-half:]

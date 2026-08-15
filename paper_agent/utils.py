"""通用工具：JSON 修复解析、限流重试、文本截断、章节标题启发式。"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, Callable

logger = logging.getLogger("paper_agent")


# ---------- JSON 解析 ----------

def extract_json(text: str) -> Any | None:
    """四级修复式 JSON 解析：剥 fence → 整体 loads → 平衡括号切片 → 去尾逗号/控制字符。"""
    if not text:
        return None

    # ① 剥离 markdown 代码块围栏
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
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


# ---------- 重试 ----------

def retry_on_rate_limit(fn: Callable, retries: int = 3, base_delay: float = 2.0):
    """429/5xx/网络错误指数退避重试。非重试性异常直接抛出。"""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 —— 分类后决定是否重试
            if attempt >= retries or not _is_retryable(e):
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            logger.warning("请求失败（%s），%.1fs 后第 %d/%d 次重试", e, delay, attempt + 1, retries)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def _is_retryable(e: Exception) -> bool:
    name = type(e).__module__ + "." + type(e).__name__
    # anthropic SDK 类型异常
    if "anthropic" in name:
        retryable = (
            "RateLimitError" in name
            or "APIConnectionError" in name
            or "InternalServerError" in name
        )
        # 5xx APIStatusError
        if "APIStatusError" in name and getattr(e, "status_code", 0) >= 500:
            retryable = True
        return retryable
    # requests 异常
    if "requests" in name:
        return "ConnectionError" in name or "Timeout" in name or "HTTPError" in name
    return False


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


_HEADING_PATTERNS = [
    r"^\s*(?:第[一二三四五六七八九十\d]+[章节部分]|[0-9]+(?:\.[0-9]+)*)\s*[^\n]{0,40}$",
    r"^\s*(?:Abstract|Introduction|Related Work|Method|Experiments?|Results?|Conclusion|References|Acknowledgments?)\b[^\n]{0,40}$",
    r"^\s*[A-Z][A-Za-z ]{2,50}\n=+$",
]


def detect_heading(line: str) -> str | None:
    """启发式判断一行是否为章节标题。"""
    line = line.strip()
    if not line or len(line) > 60:
        return None
    if re.match(r"^\s*(?:第[一二三四五六七八九十\d]+[章节部分]|[0-9]+(?:\.[0-9]+)*)\s*\S", line):
        return line
    if re.match(r"^\s*(?:Abstract|Introduction|Related Work|Method|Experiments?|Results?|Conclusion|References)\b", line, re.I):
        return line
    return None

"""LLM 封装：ChatOpenAI（DeepSeek 原生 OpenAI 兼容接口）+ JSON 修复重试。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import load_settings
from .utils import extract_json

logger = logging.getLogger("paper_agent")

JSON_INSTRUCTION = (
    "\n\n【输出要求】只输出 JSON，不要 markdown 代码块，不要任何解释性文字；"
    "禁止尾逗号和注释；字符串内不要换行。"
)


def get_llm(max_tokens: int | None = None, temperature: float | None = None) -> ChatOpenAI:
    """ChatOpenAI 指向 DeepSeek 原生接口（配置来自 .env 的 DEEPSEEK_* / ANTHROPIC_AUTH_TOKEN）。"""
    settings = load_settings()
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "base_url": settings.base_url,
        "api_key": settings.api_key,
    }
    kwargs["max_tokens"] = max_tokens or settings.max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


def complete(
    prompt: str,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """单轮文本生成。"""
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    resp = get_llm(max_tokens=max_tokens, temperature=temperature).invoke(messages)
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def complete_json(
    prompt: str,
    validator: Callable[[Any], str | None] | None = None,
    retries: int = 2,
    max_tokens: int | None = None,
    system: str | None = None,
) -> tuple[Any | None, str]:
    """生成 JSON 并经「修复解析 → 校验 → 错误回灌重试」三级兜底。

    validator: 接收解析后的数据，返回错误描述字符串或 None（合法）。
    最终仍失败返回 (None, 最后一次原始输出)。
    """
    settings = load_settings()
    llm = get_llm(max_tokens=max_tokens, temperature=settings.json_temperature)
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt + JSON_INSTRUCTION))

    last_text = ""
    for attempt in range(retries + 1):
        resp = llm.invoke(messages)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        last_text = text
        data = extract_json(text)
        if data is not None:
            err = validator(data) if validator else None
            if err is None:
                return data, text
        else:
            err = "输出不是合法 JSON"

        if attempt < retries:
            logger.warning("JSON 解析/校验失败（第 %d/%d 次）：%s", attempt + 1, retries, err[:200])
            messages.append(AIMessage(content=text))
            messages.append(
                HumanMessage(
                    content=f"你上一次的输出无法解析或不符合要求（错误：{err[:300]}）。"
                    "请重新只输出符合要求的 JSON。"
                )
            )
    logger.error("JSON 生成在 %d 次重试后仍失败", retries)
    return None, last_text

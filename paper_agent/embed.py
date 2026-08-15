"""本地 embedding 封装（LangChain 生态 HuggingFaceEmbeddings）：bge-small-zh-v1.5 懒加载，不可用时降级。"""

from __future__ import annotations

import logging

import numpy as np

from .config import load_settings
from .db import set_meta

logger = logging.getLogger("paper_agent")


class Embedder:
    """懒加载单例。加载失败时 available=False，检索自动降级纯 BM25。"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or load_settings().embed_model
        self._model = None
        self._available: bool | None = None  # None = 未尝试加载

    def load(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            logger.info(
                "正在加载 embedding 模型 %s（首次会从 HuggingFace 下载，约 100MB）…",
                self.model_name,
            )
            self._model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                encode_kwargs={"normalize_embeddings": True},
                show_progress=False,
            )
            # 触发一次真实加载，尽早暴露下载/加载错误
            self._model.embed_query("预热")
            self._available = True
            logger.info("embedding 模型加载完成")
        except Exception as e:  # noqa: BLE001 —— 任何失败都降级为纯 BM25
            logger.warning("embedding 模型不可用，检索将降级为纯 BM25：%s", e)
            self._available = False
            set_meta("embedder_ok", "0")
        if self._available:
            set_meta("embedder_ok", "1")
        return self._available

    @property
    def available(self) -> bool:
        return self.load()

    def embed_many(self, texts: list[str]) -> np.ndarray | None:
        """批量嵌入（已 L2 归一化）；不可用或输入为空返回 None。"""
        if not texts or not self.load():
            return None
        vecs = self._model.embed_documents(texts)
        return np.asarray(vecs, dtype=np.float32)


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder

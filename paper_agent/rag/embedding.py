"""本地 embedding（bge-small-zh-v1.5，懒加载）；不可用时检索自动降级纯 BM25。"""

from __future__ import annotations

import logging

import numpy as np

from ..db import set_meta
from .config import load_rag_config

logger = logging.getLogger("paper_agent")


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or load_rag_config().embedding.model
        self._model = None
        self._available: bool | None = None  # None = 尚未尝试加载

    def load(self) -> bool:
        if self._available is not None:
            return self._available
        if not load_rag_config().embedding.enabled:
            # 未启用：不加载模型、不联网，检索降级纯 BM25
            self._available = False
            set_meta("embedder_ok", "0")
            return False
        from langchain_huggingface import HuggingFaceEmbeddings

        def _build(local_only: bool) -> HuggingFaceEmbeddings:
            return HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"local_files_only": local_only},
                encode_kwargs={"normalize_embeddings": True},
                show_progress=False,
            )

        # 优先离线加载（模型已缓存时无需联网，避免网络挂起阻塞）；
        # 本地缓存未命中再尝试联网下载，仍未成功才降级纯 BM25。
        try:
            logger.info("加载 embedding 模型 %s（本地缓存）…", self.model_name)
            self._model = _build(local_only=True)
        except Exception:  # noqa: BLE001
            logger.info("本地缓存未命中，联网下载 embedding 模型…")
            try:
                self._model = _build(local_only=False)
            except Exception as e:  # noqa: BLE001 —— 任何失败都降级为纯 BM25
                logger.warning("embedding 模型不可用，检索将降级为纯 BM25：%s", e)
                self._available = False
                set_meta("embedder_ok", "0")
                return False
        self._model.embed_query("预热")  # 尽早暴露加载错误
        self._available = True
        set_meta("embedder_ok", "1")
        logger.info("embedding 模型加载完成")
        return True

    @property
    def available(self) -> bool:
        return self.load()

    def embed_many(self, texts: list[str]) -> np.ndarray | None:
        """批量嵌入（已 L2 归一化）。"""
        if not texts or not self.load():
            return None
        return np.asarray(self._model.embed_documents(texts), dtype=np.float32)


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder

"""RAG 配置：从 configs/rag.yaml 加载，可用环境变量 RAG_CONFIG 覆盖配置文件路径。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "rag.yaml"
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def _config_path() -> Path:
    return Path(os.environ.get("RAG_CONFIG", DEFAULT_CONFIG_PATH))


@dataclass
class MilvusConfig:
    uri: str = "data/milvus.db"
    collection: str = "paper_chunks"
    dim: int = 512

    def resolved_uri(self) -> str:
        """相对路径基于项目根目录解析。"""
        if self.uri.startswith(("http://", "https://", "tcp://")):
            return self.uri
        return str((ROOT_DIR / self.uri).resolve())


@dataclass
class EmbeddingConfig:
    enabled: bool = False  # 默认不启用：关闭时检索为纯 BM25，无需下载模型
    model: str = "BAAI/bge-small-zh-v1.5"
    batch_size: int = 32


@dataclass
class ChunkingConfig:
    size: int = 800
    overlap: int = 100
    content_max_length: int = 4096


@dataclass
class RetrievalConfig:
    top_k: int = 5
    max_per_paper: int = 2
    candidate_scale: int = 5


@dataclass
class McpConfig:
    name: str = "paper-rag"
    transport: str = "stdio"  # stdio | sse
    host: str = "127.0.0.1"
    port: int = 8760


@dataclass
class RagConfig:
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    mcp: McpConfig = field(default_factory=McpConfig)


@lru_cache(maxsize=1)
def load_rag_config() -> RagConfig:
    """加载 rag.yaml（进程内缓存；修改配置需重启）。"""
    path = _config_path()
    raw: dict = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _sub(cls, key: str):
        return cls(**{k: v for k, v in (raw.get(key) or {}).items() if k in cls.__dataclass_fields__})

    return RagConfig(
        milvus=_sub(MilvusConfig, "milvus"),
        embedding=_sub(EmbeddingConfig, "embedding"),
        chunking=_sub(ChunkingConfig, "chunking"),
        retrieval=_sub(RetrievalConfig, "retrieval"),
        mcp=_sub(McpConfig, "mcp"),
    )

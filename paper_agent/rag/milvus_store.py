"""Milvus 存储封装：BM25 倒排索引（原生稀疏向量）+ 可选向量索引。

- 本地 Lite 模式：uri 为文件路径（如 data/milvus.db），零依赖服务；
- 服务器模式：uri 为 http://host:19530。
- 分块入库时由 Milvus BM25 Function 自动分词生成稀疏向量；
  embedding.enabled=true 时额外写入稠密向量并建索引。
所有配置来自 configs/rag.yaml。
"""

from __future__ import annotations

import logging

from .config import MilvusConfig, load_rag_config

logger = logging.getLogger("paper_agent")


class MilvusStore:
    def __init__(self, config: MilvusConfig | None = None):
        self.config = config or load_rag_config().milvus
        self._max_content = load_rag_config().chunking.content_max_length
        self._client = None
        self._available: bool | None = None

    # ---------- 连接 ----------

    def connect(self):
        if self._client is not None:
            return self._client
        from pymilvus import MilvusClient

        self._client = MilvusClient(uri=self.config.resolved_uri())
        return self._client

    @property
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            self.connect().list_collections()  # 触发真实连接
            self._available = True
        except Exception as e:  # noqa: BLE001
            logger.warning("Milvus 不可用：%s", e)
            self._available = False
        return self._available

    def collection_exists(self) -> bool:
        try:
            return self.config.collection in self.connect().list_collections()
        except Exception:  # noqa: BLE001
            return False

    # ---------- 集合管理 ----------

    def ensure_collection(self) -> None:
        """创建集合（不存在时）。schema 与 embedding 配置不一致时重建（丢弃旧索引，重跑入库即可）。"""
        client = self.connect()
        if self.collection_exists():
            if self._schema_matches():
                return
            logger.info("集合 schema 与配置不一致（embedding 开关变化），重建集合")
            client.drop_collection(self.config.collection)
        self._create_collection()
        self._create_indexes()

    def _schema_matches(self) -> bool:
        want_dense = load_rag_config().embedding.enabled
        info = self.connect().describe_collection(self.config.collection)
        names = {f["name"] for f in info["fields"]}
        return ("embedding" in names) == want_dense and "sparse" in names

    def _create_collection(self) -> None:
        from pymilvus import DataType, Function, FunctionType, MilvusClient

        cfg = load_rag_config()
        client = self.connect()
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("paper_id", DataType.INT64)
        schema.add_field("seq", DataType.INT64)
        schema.add_field("heading", DataType.VARCHAR, max_length=256)
        schema.add_field("content", DataType.VARCHAR, max_length=self._max_content, enable_analyzer=True)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        # BM25：入库时自动对 content 分词生成稀疏向量
        schema.add_function(
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["content"],
                output_field_names="sparse",
            )
        )
        if cfg.embedding.enabled:
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.config.dim)
        client.create_collection(self.config.collection, schema=schema)
        logger.info("Milvus 集合 %s 已创建（BM25 倒排%s）", self.config.collection, "+稠密向量" if cfg.embedding.enabled else "")

    def _create_indexes(self) -> None:
        from pymilvus.milvus_client import IndexParams

        client = self.connect()
        cfg = load_rag_config()
        # BM25 稀疏倒排索引
        params = IndexParams()
        params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        client.create_index(self.config.collection, index_params=params)
        # 稠密向量索引（仅启用 embedding 时）
        if cfg.embedding.enabled:
            try:
                params = IndexParams()
                params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
                client.create_index(self.config.collection, index_params=params)
            except Exception:  # noqa: BLE001 —— 个别 Lite 版本不支持 AUTOINDEX
                params = IndexParams()
                params.add_index(field_name="embedding", index_type="FLAT", metric_type="COSINE")
                client.create_index(self.config.collection, index_params=params)
        # 标量索引（Lite 可能不支持，失败仅告警）
        try:
            params = IndexParams()
            params.add_index(field_name="paper_id", index_type="INVERTED")
            client.create_index(self.config.collection, index_params=params)
        except Exception as e:  # noqa: BLE001
            logger.debug("paper_id 标量索引创建失败（可忽略）：%s", e)

    # ---------- 写入 ----------

    def insert_chunks(self, rows: list[dict]) -> int:
        """批量插入分块。rows: [{paper_id, seq, heading, content, embedding?}]
        sparse 由 BM25 Function 自动生成；embedding 仅在启用向量时提供。"""
        if not rows:
            return 0
        self.ensure_collection()
        cleaned = [
            {
                "paper_id": int(r["paper_id"]),
                "seq": int(r["seq"]),
                "heading": (r.get("heading") or "")[:256],
                "content": (r["content"] or "")[: self._max_content],
                **({"embedding": r["embedding"]} if "embedding" in r else {}),
            }
            for r in rows
        ]
        self.connect().insert(self.config.collection, data=cleaned)
        return len(rows)

    def delete_by_paper(self, paper_id: int) -> int:
        """删除某论文的全部分块（按过滤条件），返回删除条数。"""
        self.ensure_collection()
        client = self.connect()
        before = self.count_by_paper(paper_id)
        client.delete(self.config.collection, filter=f"paper_id == {int(paper_id)}")
        return before

    # ---------- 读取 ----------

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        stats = self.connect().get_collection_stats(self.config.collection)
        return int(stats.get("row_count", 0))

    def count_by_paper(self, paper_id: int) -> int:
        if not self.collection_exists():
            return 0
        return len(
            self.connect().query(
                self.config.collection,
                filter=f"paper_id == {int(paper_id)}",
                output_fields=["id"],
                limit=16384,
            )
        )

    def paper_ids(self) -> list[int]:
        if not self.collection_exists():
            return []
        rows = self.connect().query(
            self.config.collection, filter="", output_fields=["paper_id"], limit=16384
        )
        return sorted({int(r["paper_id"]) for r in rows})

    # ---------- 检索 ----------

    def search_bm25(self, query: str, top_k: int, paper_id: int | None = None) -> list[dict]:
        """BM25 倒排检索（Milvus 原生稀疏向量），返回 [{paper_id, seq, heading, content, score}]。"""
        self.ensure_collection()
        filter_expr = f"paper_id == {int(paper_id)}" if paper_id is not None else ""
        hits = self.connect().search(
            self.config.collection,
            data=[query],
            anns_field="sparse",
            limit=top_k,
            filter=filter_expr,
            output_fields=["paper_id", "seq", "heading", "content"],
            search_params={"metric_type": "BM25", "params": {}},
        )[0]
        return [
            {
                "paper_id": h["entity"]["paper_id"],
                "seq": h["entity"]["seq"],
                "heading": h["entity"].get("heading"),
                "content": h["entity"]["content"],
                "score": float(h["distance"]),
            }
            for h in hits
        ]

    def search_dense(self, query_vec, top_k: int, paper_id: int | None = None) -> list[dict]:
        """稠密向量检索（COSINE，仅 embedding 启用时可用）。"""
        self.ensure_collection()
        filter_expr = f"paper_id == {int(paper_id)}" if paper_id is not None else ""
        hits = self.connect().search(
            self.config.collection,
            data=[list(query_vec)],
            anns_field="embedding",
            limit=top_k,
            filter=filter_expr,
            output_fields=["paper_id", "seq", "heading", "content"],
            search_params={"metric_type": "COSINE", "params": {}},
        )[0]
        return [
            {
                "paper_id": h["entity"]["paper_id"],
                "seq": h["entity"]["seq"],
                "heading": h["entity"].get("heading"),
                "content": h["entity"]["content"],
                "score": float(h["distance"]),
            }
            for h in hits
        ]

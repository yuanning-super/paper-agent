"""Milvus 存储封装：连接（Lite 本地 / 服务器）、集合管理、分块增删查、向量检索。"""

from __future__ import annotations

import logging

from .config import MilvusConfig, load_rag_config

logger = logging.getLogger("paper_agent")


class MilvusStore:
    """Milvus 客户端封装。

    - 本地 Lite 模式：uri 为文件路径（如 data/milvus.db），零依赖服务；
    - 服务器模式：uri 为 http://host:19530。
    所有配置来自 configs/rag.yaml。
    """

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
            client = self.connect()
            client.list_collections()  # 触发真实连接
            self._available = True
        except Exception as e:  # noqa: BLE001
            logger.warning("Milvus 不可用（%s），检索将降级为纯 BM25", e)
            self._available = False
        return self._available

    def collection_exists(self) -> bool:
        try:
            return self.config.collection in self.connect().list_collections()
        except Exception:  # noqa: BLE001
            return False

    # ---------- 集合管理 ----------

    def ensure_collection(self) -> None:
        """创建集合（不存在时）并建向量索引。"""
        from pymilvus import DataType, MilvusClient
        from pymilvus.milvus_client import IndexParams

        client = self.connect()
        if self.collection_exists():
            return
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("paper_id", DataType.INT64)
        schema.add_field("seq", DataType.INT64)
        schema.add_field("heading", DataType.VARCHAR, max_length=256)
        schema.add_field(
            "content", DataType.VARCHAR, max_length=self._max_content
        )
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.config.dim)
        client.create_collection(self.config.collection, schema=schema)
        logger.info("Milvus 集合 %s 已创建", self.config.collection)

        # 向量索引：优先 AUTOINDEX，个别 Lite 版本不支持时退回 FLAT
        # （pymilvus 3.x 要求 IndexParams 对象而非 dict）
        try:
            params = IndexParams()
            params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
            client.create_index(self.config.collection, index_params=params)
        except Exception:  # noqa: BLE001
            params = IndexParams()
            params.add_index(field_name="embedding", index_type="FLAT", metric_type="COSINE")
            client.create_index(self.config.collection, index_params=params)
        # 标量索引（加速按 paper_id 过滤；Lite 可能不支持，失败仅告警）
        try:
            params = IndexParams()
            params.add_index(field_name="paper_id", index_type="INVERTED")
            client.create_index(self.config.collection, index_params=params)
        except Exception as e:  # noqa: BLE001
            logger.debug("paper_id 标量索引创建失败（可忽略）：%s", e)

    # ---------- 写入 ----------

    def insert_chunks(self, rows: list[dict]) -> int:
        """批量插入分块。rows: [{paper_id, seq, heading, content, embedding(ndarray/list)}]"""
        if not rows:
            return 0
        self.ensure_collection()
        cleaned = [
            {
                "paper_id": int(r["paper_id"]),
                "seq": int(r["seq"]),
                "heading": (r.get("heading") or "")[:256],
                "content": (r["content"] or "")[: self._max_content],
                "embedding": r["embedding"],
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

    def get_chunks(self, paper_id: int) -> list[dict]:
        """某论文的全部分块（按 seq 排序，不含向量）。"""
        if not self.collection_exists():
            return []
        rows = self.connect().query(
            self.config.collection,
            filter=f"paper_id == {int(paper_id)}",
            output_fields=["id", "paper_id", "seq", "heading", "content"],
            limit=16384,
        )
        rows.sort(key=lambda r: r["seq"])
        return rows

    def fetch_all(self) -> list[dict]:
        """分页取出全库分块（用于构建 BM25 索引），按 (paper_id, seq) 排序。"""
        if not self.collection_exists():
            return []
        client = self.connect()
        page = 4096
        out: list[dict] = []
        offset = 0
        while True:
            rows = client.query(
                self.config.collection,
                filter="",
                output_fields=["id", "paper_id", "seq", "heading", "content"],
                limit=page,
                offset=offset,
            )
            out.extend(rows)
            if len(rows) < page:
                break
            offset += page
        out.sort(key=lambda r: (r["paper_id"], r["seq"]))
        return out

    def paper_ids(self) -> list[int]:
        if not self.collection_exists():
            return []
        rows = self.connect().query(
            self.config.collection,
            filter="",
            output_fields=["paper_id"],
            limit=16384,
        )
        return sorted({int(r["paper_id"]) for r in rows})

    # ---------- 检索 ----------

    def search(self, query_vec, top_k: int, paper_id: int | None = None) -> list[dict]:
        """向量检索（COSINE），返回 [{id, paper_id, seq, heading, content, score}]。"""
        self.ensure_collection()
        filter_expr = f"paper_id == {int(paper_id)}" if paper_id is not None else ""
        hits = self.connect().search(
            self.config.collection,
            data=[list(query_vec)],
            limit=top_k,
            filter=filter_expr,
            output_fields=["paper_id", "seq", "heading", "content"],
            search_params={"metric_type": "COSINE", "params": {}},
        )[0]
        return [
            {
                "id": h["id"],
                "paper_id": h["entity"]["paper_id"],
                "seq": h["entity"]["seq"],
                "heading": h["entity"].get("heading"),
                "content": h["entity"]["content"],
                "score": float(h["distance"]),
            }
            for h in hits
        ]

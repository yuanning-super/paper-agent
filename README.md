# 科研论文管理 Agent 📚

基于 **LangChain + LangGraph + Milvus** 的科研论文管理助手，采用「**工作流 + agent loop**」组合架构：

- 流程清晰的环节（入库、报告生成、创新点抽取/想法生成、检索）用 LangGraph 确定性工作流；
- 需要探索决策的环节（GitHub 仓库查找与分析、追问式问答）用 prebuilt agent loop；
- RAG（文档解析、分块、建索引、混合检索）基于 **Milvus** 向量库，位于 `paper_agent/rag/`，并作为 **MCP 工具**暴露给外部 Agent。

## 功能

1. **论文解读报告**：上传 PDF 或输入 arXiv ID → 自动入库 → 按章节拆分论文（节省上下文）→ **强制委派四个子 agent**：背景调研（数据集与 baseline）、方法分析（创新性与实现，可读官方代码）、实验分析、报告综合 → 输出 6 章节中文报告（一句话总结 / **直观方法介绍（一眼看懂）** / 方法深度分析 / 实验背景 / 实验结果与分析 / 代码仓库分析）；自动查找并浅克隆分析论文的 GitHub 仓库
2. **创新点抽取与推荐**：从论文抽取结构化创新点（方法 / 动机 / 发现 / 设计 / 数据），跨论文组合生成 3-5 个新研究想法（含可行性分析、风险、实验设计建议）
3. **论文库整理与检索问答**：元数据入库、中文关键词与研究方向自动分类、去重；检索基于 **Milvus BM25 倒排索引**（启用 embedding 时与稠密向量做 RRF 混合融合），回答带引用来源；支持**增量更新**与**删除**操作

## RAG（Milvus + MCP）

文档解析 → 分块 → 嵌入 → Milvus 建索引 → 混合检索，全部在 `paper_agent/rag/` 中：

| 文件 | 职责 |
|---|---|
| `configs/rag.yaml` | RAG 全部配置（Milvus 地址/集合、embedding 模型、分块与检索参数、MCP 服务器） |
| `rag/milvus_store.py` | Milvus 封装：BM25 倒排索引（原生稀疏向量）+ 可选稠密向量；本地 Lite 或服务器模式（URI 可配） |
| `rag/pipeline.py` | 解析/分块/建索引流水线：`index_paper`（增量）、`index_missing`（批量增量）、`remove_paper`（删除） |
| `rag/retriever.py` | 检索：Milvus BM25 倒排；embedding 启用时与稠密向量 RRF 融合 |
| `rag/mcp_server.py` | MCP 工具服务器：`rag_search` / `rag_add_paper` / `rag_update_paper` / `rag_delete_paper` / `rag_index_missing` / `rag_list_papers` / `rag_status` |

**注册到 Claude Code：**

```bash
claude mcp add paper-rag -- uv run python -m paper_agent.rag.mcp_server
```

之后在任何 Claude Code 会话中可直接调用上述 `rag_*` 工具检索/管理你的论文库。MCP 传输方式（stdio / sse）与端口在 `configs/rag.yaml` 的 `mcp` 段配置。

**Milvus 两种模式（配置文件 `milvus.uri`）：**
- 本地 Lite（默认）：`uri: data/milvus.db` —— 数据存本地文件，无需部署服务
- 服务器：`uri: http://localhost:19530` —— 连接 Milvus 服务（docker 部署 `milvusdb/milvus`）

**增量更新与删除：**
- 新增论文入库时自动分块、嵌入、写入 Milvus（增量，无需重建全库索引）
- `rag_update_paper(paper_id)`：删除旧向量后重建单篇索引
- `rag_delete_paper(paper_id)`：同时清理 Milvus 向量与库内元数据（Web 界面「论文库」页也有删除入口）
- `rag_index_missing()`：把库中未建索引的论文批量补齐（老数据迁移入口）

## 快速开始

```bash
uv sync

# 配置 .env（复制 .env.example 填入 DeepSeek Anthropic 兼容接口凭据）
cp .env.example .env

# （可选）预下载本地 embedding 模型（约 100MB）
uv run python scripts/download_embed.py

# 启动 Web 界面
uv run python paper_agent/app.py   # 或 uv run uvicorn paper_agent.app:app --port 8501
```

浏览器打开 http://127.0.0.1:8501，四个页面：论文解读 / 论文库 / 创新工坊 / 问答。同时提供 REST API（/api/*，见 paper_agent/app.py）。

## 环境变量（.env）

```bash
# DeepSeek 原生 OpenAI 兼容接口（应用主配置）
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_API_KEY=sk-你的密钥
DEEPSEEK_MODEL=deepseek-v4-pro
# 可选：GITHUB_TOKEN=ghp_xxx（提升 GitHub API 限流）
```

> 如未设置 `DEEPSEEK_API_KEY`，会回退读取 `ANTHROPIC_AUTH_TOKEN`（并自动去掉 `[1m]` 等模型名后缀）。

## 数据存储

```
configs/
└── rag.yaml       # RAG 配置（Milvus / embedding / 分块 / 检索 / MCP）
data/
├── library.db     # SQLite：论文元数据、分块文本缓存、创新点、研究想法、问答日志
├── milvus.db      # Milvus Lite 本地向量库（分块 + 向量索引，模式可配）
├── pdfs/          # 论文 PDF
├── reports/       # 解读报告 Markdown
├── cache/         # arXiv API 缓存
├── clones/        # GitHub 临时浅克隆（分析后自动清理）
└── checkpoints/   # LangGraph 线程状态（支持刷新后断点续跑）
```

## 架构

```
paper_agent/
├── config.py       # 全局配置
├── db.py           # SQLite 持久化
├── schemas.py      # Pydantic 数据模型
├── llm.py          # ChatOpenAI 封装（DeepSeek）+ JSON 修复重试
├── ingestion.py    # arXiv 抓取 / PDF 提取 / 分块 / 元数据增强
├── retrieval.py    # 检索入口（委托 rag 模块）
├── tools.py        # agent 工具集（arXiv / GitHub / 论文库）
├── prompts.py      # 中文 prompt 模板
├── rag/            # RAG 模块（自包含：分块/嵌入/Milvus 索引/检索/MCP）
│   ├── config.py        # rag.yaml 加载器
│   ├── chunking.py      # 文档分块（字符窗口 + 章节标题标注）
│   ├── embedding.py     # bge-small-zh-v1.5 本地 embedding（懒加载，不可用降级）
│   ├── milvus_store.py  # Milvus 连接/集合/增删查/向量检索
│   ├── pipeline.py      # 分块→嵌入→建索引（增量/更新/删除/迁移）
│   ├── retriever.py     # Milvus 向量 + BM25 混合检索（自动降级）
│   └── mcp_server.py    # MCP 工具服务器（mcp SDK MCPServer）
├── graphs/         # LangGraph 工作流
│   ├── ingest_graph.py     # 入库工作流（含写入 Milvus）
│   ├── interpret_graph.py  # 解读：章节拆分 + 强制委派四个子 agent
│   ├── innovate_graph.py   # 创新点抽取 + 想法生成
│   └── qa_graph.py         # 检索 + 问答（agent 可追问检索）
└── app.py          # Streamlit 界面
```

## 端到端验证

```bash
uv run python scripts/e2e.py            # 全流程（消耗真实 API token）
uv run python scripts/e2e.py --skip-llm # 只跑入库/检索/去重等离线步骤
```

## 说明

- LLM 走 `.env` 中配置的 DeepSeek 原生 OpenAI 兼容接口（langchain-openai），支持工具调用与流式输出
- **本地向量默认关闭**（`configs/rag.yaml` 的 `embedding.enabled: false`）：检索为 Milvus BM25 倒排索引，零模型依赖。需要语义检索时设为 `true` 并运行 `uv run python scripts/download_embed.py` 预下载 bge-small-zh-v1.5（约 100MB）
- embedding 模型不可用（离线/未下载）时自动降级为纯 BM25 检索
- 扫描版 PDF 无法提取文本时会明确提示，不会产生空报告
- 重复入库自动去重（按 arXiv ID；上传 PDF 按标题）

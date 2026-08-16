# 系统架构

## 1. 整体分层架构

```mermaid
flowchart TB
    subgraph FE["① 前端（app.py 内嵌单页 · 深色科技风）"]
        UI["论文解读 / 创新工坊 / 检索问答 / 论文库<br/>步骤指示器 · 实时日志 · 子agent面板 · KaTeX · 图表画廊"]
    end

    subgraph API["② FastAPI 层"]
        SSE["SSE 流式<br/>/api/interpret/stream · /api/qa/stream"]
        REST["REST<br/>ingest · papers · report · innovations · ideas · figures · rag"]
    end

    subgraph WF["③ LangGraph 工作流层（graphs/，工作流 + agent loop 组合）"]
        direction LR
        ING["ingest_graph 入库<br/>fetch→…→index"]
        INT["interpret_graph 解读<br/>load → 三子agent并行 → report"]
        QA["qa_graph 问答<br/>retrieve → answer"]
        INN["innovate_graph 创新工坊<br/>extract → generate"]
    end

    subgraph TOOLS["④ Agent 工具层（tools.py，12 个工具）"]
        T["read_section · read_repo_file · grep_repo(ripgrep) · list_repo_dir<br/>analyze_github_repo · find_github_url · web_search(DuckDuckGo)<br/>search_library · get_paper_summary · ingest 等"]
    end

    subgraph RAG["④ RAG 模块（rag/，独立文件夹 · 以 MCP 工具对外暴露）"]
        direction LR
        CH["chunking 分块/章节拆分"]
        EM["embedding 可选向量（默认关）"]
        MS["milvus_store 原生 BM25 倒排"]
        PL["pipeline 增量/更新/删除"]
        RT["retriever BM25 + RRF 融合"]
        MC["mcp_server 7 个 MCP 工具（stdio）"]
    end

    subgraph DATA["⑤ 数据层（data/）"]
        direction LR
        DB[("SQLite<br/>papers/chunks/innovations/ideas/qa_log")]
        MV[("Milvus<br/>BM25 倒排索引")]
        FS[("文件系统<br/>pdfs/ · workspaces/{id}/ · clones/")]
    end

    LLM["⑥ LLM：DeepSeek（OpenAI 兼容）"]
    EXT["⑥ 外部：arXiv API · GitHub 仓库 · DuckDuckGo 搜索"]

    UI <--> SSE
    UI <--> REST
    SSE --> WF
    REST --> WF
    WF --> TOOLS
    WF --> RAG
    WF --> LLM
    TOOLS --> EXT
    RAG --> MV
    RAG --> DB
    TOOLS --> FS
    WF --> DB
```

## 2. 论文工作区（每篇论文独立）

论文代码与全部输出统一放在自己的工作区，删除论文时级联清理：

```
data/workspaces/{paper_id}/
  repo/             # 论文代码克隆（analyze/read/grep/list 共用一份，不重复克隆）
  figures/          # 原文图表 PNG
  report.md         # 最终解读报告
  background.md     # 背景调研阶段结果
  method.md         # 方法分析阶段结果
  experiment.md     # 实验分析阶段结果
```

## 3. 四大功能工作流总览

```mermaid
flowchart LR
    subgraph W1["① 入库（ingest_graph，7 节点，出错短路）"]
        direction TB
        A1["arXiv ID / PDF 上传"] --> A2["fetch 元数据"] --> A3["download · extract 文本"] --> A4["chunk 分块"] --> A5["embed：Milvus BM25 索引"] --> A6["enrich：LLM 元数据增强"] --> A7["index：入库完成"]
    end
    subgraph W2["② 解读（interpret_graph，5 节点）"]
        direction TB
        B1["load：定位论文 + 章节拆分 + 图表提取 + 仓库分析"] --> B2["background ∥ method ∥ experiment<br/>三子agent并行"] --> B3["report：综合报告"] --> B4["工作区落盘（report.md + 三份阶段结果）"]
    end
    subgraph W3["③ 问答（qa_graph）"]
        direction TB
        C1["问题"] --> C2["retrieve：BM25 检索（+RRF）"] --> C3["answer：agent 可追问检索"] --> C4["回答 + 引用来源（qa_log）"]
    end
    subgraph W4["④ 创新工坊（innovate_graph）"]
        direction TB
        D1["勾选论文"] --> D2["extract：抽取创新点（JSON 校验重试）"] --> D3["generate：组合成新想法（引用硬校验）"] --> D4["想法入库"]
    end
```

## 4. 解读流水线（主图）

```mermaid
flowchart TB
    L["load 准备资料<br/>① 定位论文（短路匹配 → 定位agent → arXiv兜底）<br/>② 章节拆分写入缓存（read_section 按需读取）<br/>③ 提取原文图表 → 工作区 figures/<br/>④ find_github_url → analyze_github_repo<br/>　（克隆到工作区 repo/，产出 README/目录/依赖/核心文件摘要）"]
    L --> BG
    L --> MT
    L --> EX
    BG["子agent① 背景调研<br/>数据集与 baseline 背景（含参考文献溯源）"] --> RP
    MT["子agent② 方法分析<br/>创新性与实现（代码辅助，重要公式逐项解释）"] --> RP
    EX["子agent③ 实验分析<br/>实验设置 / 结果对比 / 消融解读"] --> RP
    RP["子agent④ 报告生成<br/>综合三份子报告与原文（read_section 回查），输出 6 章节报告"] --> OUT["工作区：report.md + background/method/experiment.md<br/>papers.status = interpreted"]
```

## 5. 每个子 agent 的工作流

每个子 agent 都是独立的 prebuilt agent loop（recursion_limit=30）：模型自主决定调用哪些工具、调用几轮，直到给出最终报告；三个分析子 agent 并行执行。

```mermaid
flowchart TB
    subgraph SA1["子agent① 背景调研（并行）"]
        direction LR
        I1["输入：摘要 + 实验/相关工作/参考文献章节"] --> L1{"agent loop"}
        L1 <--> T1["read_section 读章节<br/>web_search 联网搜索<br/>search_library 查论文库<br/>get_paper_summary"]
        L1 --> O1["输出：背景调研报告<br/>数据集 / baseline / 背景知识"]
    end
    subgraph SA2["子agent② 方法分析（并行）"]
        direction LR
        I2["输入：引言 + 方法章节 + 仓库素材"] --> L2{"agent loop"}
        L2 <--> T2["grep_repo 搜索代码（ripgrep）<br/>list_repo_dir 浏览目录<br/>read_repo_file 精读文件（按行）<br/>read_section"]
        L2 --> O2["输出：方法分析报告<br/>创新点 / 公式逐项解释 / 实现细节"]
    end
    subgraph SA3["子agent③ 实验分析（并行）"]
        direction LR
        I3["输入：实验章节"] --> L3{"agent loop"}
        L3 <--> T3["read_section 读章节"]
        L3 --> O3["输出：实验分析报告<br/>实验设置 / 关键结果 / 消融 / 解读"]
    end
    subgraph SA4["子agent④ 报告生成（收口）"]
        direction LR
        I4["输入：摘要 + 论文地图 + 三份子报告 + 仓库素材"] --> L4{"agent loop"}
        L4 <--> T4["read_section 回查原文"]
        L4 --> O4["输出：最终报告 6 章节<br/>一句话总结 / 直观方法介绍 / 方法深度分析<br/>实验背景 / 实验结果与分析 / 代码仓库分析"]
    end
```

## 分层说明

| 层 | 位置 | 职责 |
|---|---|---|
| 前端 | `app.py` 内嵌单页 | 三个功能入口；SSE 实时推送步骤事件与各子 agent 结果；KaTeX 公式、原文图表画廊 |
| FastAPI | `paper_agent/app.py` | REST + SSE 端点；静态资源（KaTeX、/workspaces 工作区） |
| 工作流 | `paper_agent/graphs/` | 四条 LangGraph 图：入库 / 解读 / 问答 / 创新工坊；确定性编排 + 子 agent loop |
| 工具 | `paper_agent/tools.py` | 12 个工具：章节按需读取、代码探索（grep/列目录/按行读）、仓库分析、联网搜索、检索等 |
| RAG | `paper_agent/rag/` | 分块与章节拆分、Milvus 原生 BM25 倒排索引、增量/删除、混合检索、MCP 服务 |
| 数据 | `data/` | SQLite + Milvus + 文件系统（PDF、每篇论文工作区、代码克隆） |

## 关键设计

- **工作流 + agent loop 组合**：主图确定性控制流程（顺序、并行、条件路由），节点内部委托 prebuilt agent loop 处理开放式任务。
- **解读流水线**：`load` 先做章节拆分（节省上下文）+ 图表提取 + 仓库分析，然后 background / method / experiment 三个子 agent **并行**执行，report agent 收口；各阶段结果与最终报告一并落盘到论文工作区。
- **代码探索工具链**：`analyze_github_repo` 克隆到论文工作区并出结构摘要；子 agent 用 `grep_repo`（ripgrep）定位实现 → `list_repo_dir` 浏览 → `read_repo_file` 按行精读，全程共用一份克隆。
- **背景调研联网**：背景调研 agent 持有 `web_search`（DuckDuckGo，免 key），可联网调研数据集与 baseline。
- **RAG 全走 Milvus**：原生 BM25 倒排索引（分析器分词 + SPARSE_FLOAT_VECTOR），embedding 默认关闭；开启时与稠密向量 RRF 融合。配置在 `configs/rag.yaml`，支持增量更新与删除，并以 MCP 工具（stdio）暴露给外部 agent。
- **配置**：全局参数在 `paper_agent/config.py`，RAG 参数在 `configs/rag.yaml`，凭据在 `.env`（不入库）。

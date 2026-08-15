"""端到端验证脚本：入库 → 解读报告 → 创新点 → 研究想法 → 检索问答。

用法：
    uv run python scripts/e2e.py            # 全流程（消耗真实 API token）
    uv run python scripts/e2e.py --skip-llm # 只跑不调用 LLM 的步骤
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = "✅ PASS"
FAIL = "❌ FAIL"

ARXIV_NO_REPO = "1706.03762"  # Attention Is All You Need（无官方仓库）
ARXIV_WITH_REPO = "2205.14135"  # FlashAttention-2（有官方 GitHub 仓库）

REPORT_SECTIONS = [
    "背景与动机",
    "核心贡献",
    "方法与技术细节",
    "实验结果",
    "局限与不足",
    "相关工作对比",
    "代码仓库分析",
    "一句话总结",
]


def check(name: str, cond: bool, detail: str = "") -> bool:
    mark = PASS if cond else FAIL
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def run(skip_llm: bool) -> int:
    import os

    if skip_llm:
        os.environ["PAPER_AGENT_SKIP_LLM"] = "1"

    from paper_agent.config import load_settings
    from paper_agent.db import get_chunks, get_meta, get_paper, init_db, list_papers
    from paper_agent.graphs import run_ingest
    from paper_agent.retrieval import search

    # 重置数据库与 Milvus，保证 e2e 可重复执行（PDF/arXiv 缓存/embedding 模型不受影响）
    settings = load_settings()
    for f in settings.data_dir.glob("library.db*"):
        f.unlink()
    from paper_agent.rag.config import load_rag_config

    milvus_uri = load_rag_config().milvus.resolved_uri()
    milvus_path = Path(milvus_uri)
    if not milvus_uri.startswith(("http://", "https://")):
        for f in milvus_path.parent.glob(milvus_path.name + "*"):
            f.unlink()
    init_db()
    failed = 0
    results: dict[str, int] = {}

    # ---------- 1. 入库：无官方仓库的论文 ----------
    print("\n=== 1. 入库", ARXIV_NO_REPO, "===")
    r1 = run_ingest(ARXIV_NO_REPO)
    ok = (
        r1.paper_id is not None
        and r1.is_new
        and r1.chunk_count > 0
        and r1.status == "enriched"
        and not r1.error
    )
    failed += not check("入库成功（papers=1, chunks>0, status=enriched）", ok, f"paper_id={r1.paper_id}, chunks={r1.chunk_count}, error={r1.error}")
    results["paper1"] = r1.paper_id or 0

    # 向量索引已写入 Milvus
    from paper_agent.rag.pipeline import status as rag_status

    st = rag_status()
    failed += not check(
        "Milvus 向量索引已建立",
        st.get("milvus_available") and st.get("vector_count", 0) > 0,
        f"milvus={st.get('milvus_available')}, vectors={st.get('vector_count')}",
    )

    # ---------- 2. 重复入库去重 ----------
    print("\n=== 2. 重复入库去重 ===")
    r2 = run_ingest(ARXIV_NO_REPO)
    ok = r2.is_new is False and r2.paper_id == r1.paper_id
    failed += not check("重复入库返回 is_new=false 且 ID 不变", ok, f"paper_id={r2.paper_id}")

    # ---------- 3. 入库：有官方仓库的论文 ----------
    print("\n=== 3. 入库", ARXIV_WITH_REPO, "===")
    r3 = run_ingest(ARXIV_WITH_REPO)
    ok = r3.paper_id is not None and r3.chunk_count > 0 and not r3.error
    failed += not check("入库成功", ok, f"paper_id={r3.paper_id}, chunks={r3.chunk_count}")
    results["paper2"] = r3.paper_id or 0

    # ---------- 4. GitHub 链接查找 ----------
    print("\n=== 4. GitHub 仓库查找 ===")
    from paper_agent.tools import find_github_url

    gh = find_github_url.invoke({"paper_id": results["paper2"]})
    import json as _json

    gh_data = _json.loads(str(gh))
    ok = gh_data.get("url") and "github.com" in gh_data["url"]
    failed += not check("找到官方仓库", ok, f"url={gh_data.get('url')}, source={gh_data.get('source')}")

    if skip_llm:
        print("\n（--skip-llm：跳过 LLM 相关步骤）")
        return failed

    # ---------- 5. 解读报告（无仓库论文） ----------
    print("\n=== 5. 解读报告：", ARXIV_NO_REPO, "===")
    from paper_agent.graphs import build_interpret_graph

    graph = build_interpret_graph()
    state = graph.invoke({"query": f"解读论文库中 ID 为 {results['paper1']} 的论文"})
    if state.get("error"):
        failed += not check("报告生成", False, state["error"])
    else:
        report_text = state.get("report_text", "")
        missing = [s for s in REPORT_SECTIONS if s not in report_text]
        failed += not check("报告包含全部 8 个章节", not missing, f"缺失：{missing}")
        p1 = get_paper(results["paper1"])
        failed += not check(
            "报告已落盘且状态更新为 interpreted",
            bool(p1 and p1.get("report_path") and Path(p1["report_path"]).exists() and p1["status"] == "interpreted"),
            f"report_path={p1.get('report_path') if p1 else None}",
        )

    # ---------- 6. 创新点抽取 ----------
    print("\n=== 6. 创新点抽取 ===")
    from paper_agent.db import get_innovations
    from paper_agent.graphs import run_extract

    state = run_extract([results["paper1"], results["paper2"]])
    innovs1 = get_innovations(results["paper1"])
    innovs2 = get_innovations(results["paper2"])
    valid_kinds = {"method", "motivation", "finding", "design", "dataset"}
    ok = bool(innovs1 and innovs2) and all(
        it["kind"] in valid_kinds for it in innovs1 + innovs2
    )
    failed += not check(
        "两篇论文均抽取成功且 kind 合法",
        ok,
        f"paper1={len(innovs1)} 条, paper2={len(innovs2)} 条, error={state.get('error')}",
    )

    # ---------- 7. 跨论文研究想法 ----------
    print("\n=== 7. 跨论文研究想法 ===")
    from paper_agent.graphs import run_ideas

    state = run_ideas([results["paper1"], results["paper2"]])
    ideas = state.get("ideas", [])
    ok = 3 <= len(ideas) <= 5
    failed += not check("想法数量 3-5", ok, f"实际 {len(ideas)} 条, error={state.get('error')}")
    if ideas:
        from paper_agent.db import list_ideas

        saved_ideas = list_ideas()
        ok = all(len(i.get("source_innovations", [])) >= 2 for i in saved_ideas[: len(ideas)])
        failed += not check("每个想法引用 ≥2 个创新点", ok)
        ok = bool(saved_ideas)
        failed += not check("想法已持久化", ok, f"ideas 表 {len(saved_ideas)} 条")

    # ---------- 8. 混合检索 ----------
    print("\n=== 8. 混合检索 ===")
    hits = search("自注意力机制", top_k=3)
    ok = bool(hits) and hits[0].paper_id == results["paper1"]
    failed += not check("语义检索命中", ok, f"top1={hits[0].title if hits else None}, score={hits[0].score if hits else None}")
    hits2 = search("注意力 高效 计算", top_k=3)
    ok = bool(hits2) and results["paper2"] in [h.paper_id for h in hits2]
    failed += not check("跨论文检索命中 FlashAttention", ok, f"papers={[h.paper_id for h in hits2]}")

    # ---------- 9. 检索问答 ----------
    print("\n=== 9. 检索问答 ===")
    from paper_agent.graphs import run_qa

    qa = run_qa("Transformer 的核心贡献是什么？")
    answer = qa.get("answer", "")
    ok = bool(answer) and not qa.get("error")
    failed += not check("生成回答", ok, f"error={qa.get('error')}")
    citations = qa.get("citations", [])
    failed += not check("回答带引用来源", bool(citations), f"citations={len(citations)} 条")

    # ---------- 10. embedding / Milvus 降级 ----------
    print("\n=== 10. Milvus 降级（纯 BM25）===")
    from paper_agent.rag import pipeline as _rag_pipeline
    from paper_agent.rag import retriever as _rag_retriever

    old_store = _rag_pipeline._store
    old_retriever = _rag_retriever._retriever

    class _NoMilvus:  # 模拟 Milvus 不可用
        available = False

    _rag_pipeline._store = _NoMilvus()
    _rag_retriever._retriever = None
    try:
        hits3 = search("transformer", top_k=3)
        ok = bool(hits3)
        failed += not check("降级后检索正常（纯 BM25）", ok, f"hits={len(hits3)}")
    finally:
        _rag_pipeline._store = old_store
        _rag_retriever._retriever = old_retriever

    # ---------- 11. MCP 工具冒烟 + 增量/更新/删除 ----------
    print("\n=== 11. MCP 工具 + 增量更新 + 删除 ===")
    from paper_agent.rag.mcp_server import (
        rag_delete_paper,
        rag_index_missing,
        rag_list_papers,
        rag_search,
        rag_status as _mcp_status,
        rag_update_paper,
    )

    r = _mcp_status()
    failed += not check("rag_status 工具可用", r.startswith("{"), "")

    r = rag_search("自注意力机制", 3)
    ok = '"paper_id"' in r and '"hits"' in r
    failed += not check("rag_search 工具返回命中", ok, r[:80])

    r = rag_index_missing()
    ok = '"ok": true' in r and '"total_indexed": 0' in r
    failed += not check("rag_index_missing 增量幂等（已全索引，无需新增）", ok, r[:120])

    r = rag_update_paper(results["paper1"])
    ok = '"ok": true' in r and '"indexed"' in r
    failed += not check("rag_update_paper 重建索引", ok, r[:120])

    r = rag_list_papers()
    ok = '"indexed": true' in r
    failed += not check("rag_list_papers 标注索引状态", ok, r[:100])

    # 删除 paper2：Milvus 向量 + 元数据一并清除，检索不可再命中
    r = rag_delete_paper(results["paper2"], remove_library=True)
    ok = '"ok": true' in r
    failed += not check("rag_delete_paper 删除成功", ok, r[:150])
    hits_after = search("FlashAttention", top_k=5)
    ok = all(h.paper_id != results["paper2"] for h in hits_after)
    failed += not check("删除后检索不再命中该论文", ok, f"hits={[h.paper_id for h in hits_after]}")
    ok = get_paper(results["paper2"]) is None
    failed += not check("论文元数据已从库中移除", ok)

    print(f"\n{'=' * 40}\n共 {failed} 项失败")
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="端到端验证")
    parser.add_argument("--skip-llm", action="store_true", help="跳过消耗 LLM token 的步骤")
    args = parser.parse_args()
    sys.exit(1 if run(args.skip_llm) else 0)

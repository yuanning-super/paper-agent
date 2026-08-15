"""Streamlit 界面：论文库 / 论文解读 / 创新点 / 研究想法 / 问答 五页导航。"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
import uuid
from pathlib import Path

import streamlit as st

from .config import load_settings
from .db import get_chunks, get_innovations, init_db, list_ideas, list_papers
from .embed import get_embedder
from .graphs import build_interpret_graph, run_extract, run_ideas
from .graphs.qa_graph import build_qa_graph
from .ingestion import ingest_pdf_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

st.set_page_config(page_title="科研论文管理 Agent", page_icon="📚", layout="wide")


# ---------- 缓存资源 ----------

@st.cache_resource
def _init() -> None:
    init_db()
    load_settings().ensure_dirs()


@st.cache_resource
def _checkpointer():
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings = load_settings()
    conn = sqlite3.connect(
        settings.checkpoints_dir / "checkpoints.sqlite", check_same_thread=False
    )
    return SqliteSaver(conn)


@st.cache_resource
def _embedder_ok() -> bool:
    return get_embedder().available


_init()

# ---------- 侧边栏 ----------

st.sidebar.title("📚 论文管理 Agent")
page = st.sidebar.radio(
    "导航",
    ["📤 论文解读", "📚 论文库", "💡 创新点", "🧪 研究想法", "🔍 问答检索"],
)

settings = load_settings()
if not _embedder_ok():
    st.sidebar.warning(
        "⚠️ 本地 embedding 模型不可用，检索将使用纯 BM25。\n"
        "可运行 `uv run python scripts/download_embed.py` 预下载后重启。"
    )


def _paper_label(p: dict) -> str:
    return f"#{p['id']} {p.get('title_zh') or p['title'][:60]}"


# ---------- 页面：论文解读 ----------

if page == "📤 论文解读":
    st.title("论文解读")
    st.caption("上传 PDF 或输入 arXiv ID，生成包含贡献、核心方法、代码仓库分析的中文解读报告。")

    tab_pdf, tab_arxiv, tab_lib = st.tabs(["上传 PDF", "arXiv 论文", "库内论文"])

    query = None
    with tab_pdf:
        uploaded = st.file_uploader("选择论文 PDF", type=["pdf"])
        if uploaded is not None:
            tmp = Path(tempfile.mkdtemp()) / uploaded.name
            tmp.write_bytes(uploaded.getbuffer())
            if st.button("解读该 PDF", type="primary"):
                query = f"pdf:{tmp}"
    with tab_arxiv:
        arxiv_id = st.text_input("arXiv ID", placeholder="如 1706.03762 / arXiv:2205.14135v2")
        if arxiv_id and st.button("解读该论文", type="primary"):
            query = f"arxiv:{arxiv_id.strip()}"
    with tab_lib:
        papers = list_papers()
        if papers:
            pick = st.selectbox("选择库内论文", options=papers, format_func=_paper_label)
            if st.button("解读选中论文", type="primary"):
                query = f"paper:{pick['id']}"

    if query:
        tid = "interpret-" + uuid.uuid4().hex[:12]
        graph = build_interpret_graph(checkpointer=_checkpointer())
        config = {"configurable": {"thread_id": tid}}

        st.write("**任务**：", query.replace("pdf:", "上传 PDF：").replace("arxiv:", "arXiv："))
        status = st.status("正在准备论文资料…", expanded=True)
        try:
            if query.startswith("pdf:"):
                pdf_path = query[4:]
                status.write("正在入库上传的 PDF…")
                result = ingest_pdf_file(pdf_path)
                for ev in result.events:
                    status.write("- " + ev)
                if result.error:
                    st.error(result.error)
                else:
                    query = f"paper:{result.paper_id}"

            placeholder = st.empty()
            text_so_far = ""
            status.write("探索阶段：入库 / 查找 GitHub 仓库 / 分析代码…")
            status.write("报告生成中…")
            for chunk in graph.stream(
                {"query": query},
                stream_mode="messages",
                config=config,
            ):
                msg, meta = chunk
                if (
                    meta.get("langgraph_node") == "report"
                    and isinstance(msg.content, str)
                    and msg.content
                ):
                    text_so_far += msg.content
                    placeholder.markdown(text_so_far)

            final = graph.get_state(config).values
            status.update(label="完成", state="complete", expanded=False)
            if final.get("error"):
                st.error(final["error"])
            else:
                st.success(f"报告已保存：{final.get('report_path')}")
                if not text_so_far:  # 兜底：流式无输出时直接展示
                    st.markdown(final.get("report_text", ""))
        except Exception as e:  # noqa: BLE001
            status.update(label="失败", state="error")
            st.error(f"解读失败：{e}")

# ---------- 页面：论文库 ----------

elif page == "📚 论文库":
    st.title("论文库")
    papers = list_papers()
    st.caption(f"共 {len(papers)} 篇论文。上传/解读后自动入库并分类整理。")

    if not papers:
        st.info("论文库为空。请到「论文解读」页上传 PDF 或输入 arXiv ID。")
    else:
        kind = st.selectbox("按研究方向筛选", ["全部"] + sorted({p.get("classification") or "未分类" for p in papers}))
        shown = papers if kind == "全部" else [p for p in papers if (p.get("classification") or "未分类") == kind]

        rows = [
            {
                "ID": p["id"],
                "标题": p.get("title_zh") or p["title"],
                "原文标题": p["title"],
                "arXiv": p.get("arxiv_id") or "—",
                "分类": p.get("classification") or "—",
                "关键词": "、".join(p.get("keywords", [])[:5]),
                "状态": p.get("status"),
                "报告": "✅" if p.get("report_path") else "—",
            }
            for p in shown
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("删除论文")
        del_pick = st.selectbox(
            "选择要删除的论文",
            options=papers,
            format_func=_paper_label,
            key="del_pick",
        )
        confirm = st.checkbox(
            f"确认删除 #{del_pick['id']}《{(del_pick.get('title_zh') or del_pick['title'])[:40]}》？"
            "（同时删除 Milvus 向量索引与库内元数据，不可恢复）"
        )
        if confirm and st.button("删除", type="secondary"):
            from .rag.pipeline import remove_paper

            r = remove_paper(del_pick["id"])
            if r.get("ok"):
                st.success(f"已删除（清理 {r.get('vectors_deleted', 0)} 块向量）")
                st.rerun()
            else:
                st.error(r.get("error", "删除失败"))

        st.divider()
        st.subheader("查看解读报告")
        with_report = [p for p in papers if p.get("report_path")]
        if not with_report:
            st.info("还没有解读报告，去「论文解读」页生成。")
        else:
            pick = st.selectbox("选择论文", options=with_report, format_func=_paper_label)
            report_path = Path(pick["report_path"])
            if report_path.exists():
                st.download_button(
                    "下载报告 (Markdown)",
                    data=report_path.read_text(encoding="utf-8"),
                    file_name=report_path.name,
                )
                st.markdown(report_path.read_text(encoding="utf-8"))
            else:
                st.warning("报告文件缺失，请重新解读。")

# ---------- 页面：创新点 ----------

elif page == "💡 创新点":
    st.title("创新点抽取")
    st.caption("从论文中抽取结构化创新点（方法/动机/发现/设计/数据），作为研究想法组合的素材。")

    papers = list_papers()
    ready = [p for p in papers if get_chunks(p["id"])]
    if not ready:
        st.info("论文库为空或尚未入库。请先到「论文解读」页入库论文。")
    else:
        picks = st.multiselect("选择论文（可多选）", options=ready, format_func=_paper_label)
        if picks and st.button("抽取创新点", type="primary"):
            with st.status("正在抽取创新点…", expanded=True) as status:
                result = run_extract([p["id"] for p in picks])
                for ev in result.get("events", []):
                    status.write("- " + ev)
                if result.get("error"):
                    st.error(result["error"])
                else:
                    status.update(label="完成", state="complete")
                    st.success("抽取完成，已保存到论文库。")

        st.divider()
        st.subheader("已抽取的创新点")
        show_paper = st.selectbox(
            "查看某篇论文的创新点",
            options=papers,
            format_func=_paper_label,
            key="innov_view",
        )
        innovations = get_innovations(show_paper["id"])
        if not innovations:
            st.info("该论文还没有抽取过创新点。")
        for it in innovations:
            kind_map = {
                "method": "方法",
                "motivation": "动机",
                "finding": "发现",
                "design": "设计",
                "dataset": "数据",
            }
            with st.expander(f"[{kind_map.get(it['kind'], it['kind'])}] {it['title']}"):
                st.markdown(f"**描述**：{it['description']}")
                st.markdown(f"**新意**：{it['novelty'] or '—'}")
                st.caption(f"溯源分块：{it.get('source_chunk_ids', [])}")

# ---------- 页面：研究想法 ----------

elif page == "🧪 研究想法":
    st.title("创新点组合与推荐")
    st.caption("选择多篇论文，将其创新点组合或拓展，生成有潜力且带可行性分析的新研究想法。")

    papers = list_papers()
    candidates = [p for p in papers if get_innovations(p["id"])]
    if len(candidates) < 2:
        st.info("需要至少 2 篇已抽取创新点的论文。请先到「创新点」页抽取。")
    else:
        picks = st.multiselect(
            "选择论文（建议跨方向，≥2 篇）", options=candidates, format_func=_paper_label
        )
        if picks and st.button("生成研究想法", type="primary"):
            with st.status("正在组合创新点…", expanded=True) as status:
                result = run_ideas([p["id"] for p in picks])
                for ev in result.get("events", []):
                    status.write("- " + ev)
                if result.get("error"):
                    st.error(result["error"])
                else:
                    status.update(label="完成", state="complete")
                    st.success("研究想法已生成并保存。")

    st.divider()
    st.subheader("历史研究想法")
    ideas = list_ideas()
    if not ideas:
        st.info("还没有生成过研究想法。")
    for idea in ideas:
        with st.expander(f"💡 {idea['title']}"):
            st.markdown(f"**核心假设**：{idea['hypothesis']}")
            st.markdown(f"**组合方式**：{idea['combination']}")
            st.markdown(f"**可行性**：{idea.get('feasibility') or '—'}")
            st.markdown("**潜在风险**：")
            for r in idea.get("risks", []):
                st.markdown(f"- {r}")
            st.markdown("**实验设计建议**：")
            for e in idea.get("experiments", []):
                st.markdown(f"- {e}")
            st.caption(f"来源创新点：{idea.get('source_innovations', [])}　|　来源论文：{idea.get('source_papers', [])}")

# ---------- 页面：问答检索 ----------

elif page == "🔍 问答检索":
    st.title("论文库问答")
    st.caption("基于混合检索（关键词 + 语义）在你的论文库中查找答案，回答带引用来源。")

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    question = st.chat_input("问点什么？例如：Transformer 的核心贡献是什么？")
    if question:
        st.session_state.qa_history.append({"role": "user", "content": question})
        tid = "qa-" + uuid.uuid4().hex[:12]
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            text_so_far = ""
            try:
                g = build_qa_graph(checkpointer=_checkpointer())
                config = {"configurable": {"thread_id": tid}}
                for chunk in g.stream(
                    {"question": question},
                    stream_mode="messages",
                    config=config,
                ):
                    msg, meta = chunk
                    if (
                        meta.get("langgraph_node") == "answer"
                        and isinstance(msg.content, str)
                        and msg.content
                    ):
                        text_so_far += msg.content
                        placeholder.markdown(text_so_far)
                final = g.get_state(config).values
                if final.get("error"):
                    st.error(final["error"])
                elif not text_so_far:
                    placeholder.markdown(final.get("answer", ""))
                citations = final.get("citations", [])
                if citations:
                    with st.expander("📎 参考来源"):
                        for c in citations:
                            st.markdown(
                                f"- 《{c.get('title', '')}》"
                                f"{'（arXiv: ' + c['arxiv_id'] + '）' if c.get('arxiv_id') else ''}"
                                f"　分块 #{c.get('chunk_seq')}"
                            )
            except Exception as e:  # noqa: BLE001
                st.error(f"问答失败：{e}")

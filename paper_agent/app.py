"""FastAPI 前端：REST API + 单页界面（论文解读 / 论文库 / 创新工坊 / 问答）。"""

from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from paper_agent.config import load_settings
from paper_agent.db import get_innovations, init_db, list_ideas, list_papers
from paper_agent.graphs import build_interpret_graph, run_extract, run_ideas
from paper_agent.graphs.qa_graph import build_qa_graph
from paper_agent.ingestion import ingest_arxiv, ingest_pdf_file, normalize_arxiv_id
from paper_agent.rag.pipeline import remove_paper
from paper_agent.rag.pipeline import status as rag_status

init_db()
load_settings().ensure_dirs()

app = FastAPI(title="科研论文管理 Agent")

# 静态资源：KaTeX（本地托管，离线可用）与论文图表
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")
app.mount("/figures", StaticFiles(directory=load_settings().figures_dir), name="figures")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------- 解读 ----------

class InterpretRequest(BaseModel):
    query: str


def _iter_interpret(query: str):
    cfg = {"configurable": {"thread_id": f"api-{uuid.uuid4().hex[:12]}"}}
    graph = build_interpret_graph()
    final: dict = {}
    # 子 agent 输出字段 → 阶段名（完成时把完整结果推给前端）
    stage_fields = {
        "background": "background",
        "method_analysis": "method",
        "experiment_analysis": "experiment",
    }
    for chunk in graph.stream({"query": query}, stream_mode="updates", config=cfg):
        for update in chunk.values():
            final.update(update)  # 累积节点状态，流结束后作为最终结果
            for ev in update.get("events", []):
                yield _sse({"event": ev})
            for field, stage in stage_fields.items():
                if update.get(field):
                    yield _sse({"stage": stage, "result": update[field]})
    yield _sse(
        {
            "done": True,
            "paper_id": final.get("paper_id"),
            "report_path": final.get("report_path"),
            "report": final.get("report_text", ""),
            "error": final.get("error"),
        }
    )


@app.post("/api/interpret/stream")
def interpret_stream(req: InterpretRequest) -> StreamingResponse:
    return StreamingResponse(_iter_interpret(req.query), media_type="text/event-stream")


# ---------- 入库 ----------

@app.post("/api/ingest/arxiv")
def ingest_arxiv_paper(arxiv_id: str) -> JSONResponse:
    result = ingest_arxiv(normalize_arxiv_id(arxiv_id))
    return JSONResponse(result.to_dict() | {"events": result.events})


@app.post("/api/ingest/pdf")
async def ingest_upload(file: UploadFile) -> JSONResponse:
    tmp = Path(tempfile.mkdtemp()) / (file.filename or "upload.pdf")
    tmp.write_bytes(await file.read())
    result = ingest_pdf_file(tmp)
    return JSONResponse(result.to_dict() | {"events": result.events})


# ---------- 论文库 ----------

@app.get("/api/papers")
def papers() -> JSONResponse:
    return JSONResponse(
        [
            {
                "id": p["id"],
                "title": p.get("title_zh") or p["title"],
                "title_orig": p["title"],
                "arxiv_id": p.get("arxiv_id"),
                "classification": p.get("classification"),
                "status": p.get("status"),
                "has_report": bool(p.get("report_path")),
            }
            for p in list_papers()
        ]
    )


@app.get("/api/papers/{paper_id}/report")
def paper_report(paper_id: int) -> JSONResponse:
    from paper_agent.db import get_paper

    paper = get_paper(paper_id)
    if not paper or not paper.get("report_path"):
        return JSONResponse({"error": "报告不存在，请先解读"}, status_code=404)
    text = Path(paper["report_path"]).read_text(encoding="utf-8")
    # 各阶段子报告（与最终报告一并保存）
    steps = {}
    for key, label in (("background", "背景调研"), ("method", "方法分析"), ("experiment", "实验分析")):
        p = load_settings().reports_dir / f"{paper_id}.{key}.md"
        if p.exists():
            steps[label] = p.read_text(encoding="utf-8")
    return JSONResponse({"paper_id": paper_id, "report": text, "steps": steps})


@app.delete("/api/papers/{paper_id}")
def delete_paper(paper_id: int) -> JSONResponse:
    return JSONResponse(remove_paper(paper_id))


# ---------- 创新工坊 ----------

class PaperIdsRequest(BaseModel):
    paper_ids: list[int]


@app.post("/api/innovations/extract")
def extract_innovations(req: PaperIdsRequest) -> JSONResponse:
    return JSONResponse(run_extract(req.paper_ids))


@app.get("/api/innovations/{paper_id}")
def innovations_of(paper_id: int) -> JSONResponse:
    return JSONResponse(get_innovations(paper_id))


@app.post("/api/ideas")
def generate_ideas(req: PaperIdsRequest) -> JSONResponse:
    return JSONResponse(run_ideas(req.paper_ids))


@app.get("/api/ideas")
def ideas() -> JSONResponse:
    return JSONResponse(list_ideas())


# ---------- 问答 ----------

class QARequest(BaseModel):
    question: str


def _iter_qa(question: str):
    cfg = {"configurable": {"thread_id": f"qa-{uuid.uuid4().hex[:12]}"}}
    graph = build_qa_graph()
    final: dict = {}
    for chunk in graph.stream({"question": question}, stream_mode="updates", config=cfg):
        for update in chunk.values():
            final.update(update)
            for ev in update.get("events", []):
                yield _sse({"event": ev})
    yield _sse(
        {"done": True, "answer": final.get("answer", ""), "citations": final.get("citations", []), "error": final.get("error")}
    )


@app.post("/api/qa/stream")
def qa_stream(req: QARequest) -> StreamingResponse:
    return StreamingResponse(_iter_qa(req.question), media_type="text/event-stream")


@app.post("/api/qa")
def qa(req: QARequest) -> JSONResponse:
    from paper_agent.graphs.qa_graph import run_qa

    state = run_qa(req.question)
    return JSONResponse(
        {"answer": state.get("answer", ""), "citations": state.get("citations", []), "error": state.get("error")}
    )


@app.get("/api/papers/{paper_id}/figures")
def paper_figures(paper_id: int) -> JSONResponse:
    """论文提取出的原文图表列表。"""
    settings = load_settings()
    out_dir = settings.figures_dir / str(paper_id)
    if not out_dir.exists():
        return JSONResponse([])
    return JSONResponse(
        [
            {"name": f.name, "url": f"/figures/{paper_id}/{f.name}", "page": f.name.split("_p")[-1].split(".")[0]}
            for f in sorted(out_dir.glob("*.png"))
        ]
    )


@app.get("/api/rag/status")
def rag_status_api() -> JSONResponse:
    return JSONResponse(rag_status())


# ---------- 单页界面 ----------

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>科研论文管理 Agent</title>
<link rel="stylesheet" href="/static/katex/katex.min.css">
<script defer src="/static/katex/katex.min.js"></script>
<script defer src="/static/katex/contrib/auto-render.min.js"></script>
<style>
  :root {
    --bg: #070b14; --bg2: #0c1322; --card: rgba(148, 180, 255, 0.045);
    --border: rgba(103, 232, 249, 0.14); --border-strong: rgba(103, 232, 249, 0.35);
    --ink: #dbe4f5; --muted: #7d8aa5;
    --cyan: #22d3ee; --violet: #8b5cf6; --ok: #34d399; --danger: #f87171;
    --glow: 0 0 18px rgba(34, 211, 238, 0.28);
  }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
         margin: 0; color: var(--ink); font-size: 14px; min-height: 100vh;
         background:
           radial-gradient(1100px 500px at 85% -10%, rgba(139, 92, 246, 0.14), transparent 60%),
           radial-gradient(900px 480px at -10% 20%, rgba(34, 211, 238, 0.10), transparent 55%),
           linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%); }
  header { padding: 26px 38px 22px; position: relative; overflow: hidden;
           border-bottom: 1px solid var(--border); backdrop-filter: blur(8px); }
  header::after { content: ''; position: absolute; inset: 0; pointer-events: none;
           background: linear-gradient(90deg, transparent, rgba(34,211,238,.06), transparent); }
  header h1 { margin: 0; font-size: 21px; letter-spacing: 1px; position: relative;
           background: linear-gradient(90deg, #67e8f9, #a78bfa 70%); -webkit-background-clip: text; background-clip: text; color: transparent; }
  header p { margin: 7px 0 0; color: var(--muted); font-size: 12.5px; letter-spacing: .6px; position: relative; }
  nav { display: flex; gap: 10px; padding: 0 38px; position: sticky; top: 0; z-index: 10;
        background: rgba(7, 11, 20, 0.82); backdrop-filter: blur(14px); border-bottom: 1px solid var(--border); }
  nav button { border: none; background: none; padding: 15px 18px 13px; font-size: 13.5px; cursor: pointer;
               color: var(--muted); letter-spacing: .4px; border-bottom: 2px solid transparent; transition: all .18s; }
  nav button:hover { color: var(--cyan); text-shadow: 0 0 12px rgba(34,211,238,.5); }
  nav button.active { color: #e0f7ff; border-bottom-color: var(--cyan); text-shadow: 0 0 14px rgba(34,211,238,.65); }
  main { max-width: 1080px; margin: 28px auto 70px; padding: 0 24px; }
  .panel { display: none; }
  .panel.active { display: block; animation: fade .3s ease; }
  @keyframes fade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; } }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 16px;
          padding: 24px 26px; margin-bottom: 20px; backdrop-filter: blur(12px);
          box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35); }
  .card:hover { border-color: var(--border-strong); transition: border .3s; }
  .card h3 { margin: 0 0 16px; font-size: 15px; letter-spacing: .5px; color: #e8f1ff; }
  .row { display: flex; gap: 12px; }
  input[type=text], input:not([type]), textarea { flex: 1; padding: 12px 15px; font-size: 13.5px;
          background: rgba(2, 6, 18, 0.65); border: 1px solid var(--border); border-radius: 10px;
          color: var(--ink); outline: none; transition: all .18s; }
  input::placeholder { color: #4d5a75; }
  input:focus { border-color: var(--cyan); box-shadow: 0 0 0 3px rgba(34,211,238,.12), var(--glow); }
  button.primary { background: linear-gradient(120deg, #0891b2, #4f46e5); color: #fff; border: none;
          padding: 12px 24px; border-radius: 10px; cursor: pointer; font-size: 13.5px; font-weight: 600;
          letter-spacing: .5px; white-space: nowrap; transition: all .2s; }
  button.primary:hover:not(:disabled) { box-shadow: var(--glow); transform: translateY(-1px); }
  button.primary:disabled { opacity: .5; cursor: wait; }
  button.ghost { background: transparent; color: var(--cyan); border: 1px solid var(--border-strong);
          padding: 11px 22px; border-radius: 10px; cursor: pointer; font-weight: 600; letter-spacing: .5px; transition: all .2s; }
  button.ghost:hover { background: rgba(34,211,238,.08); box-shadow: var(--glow); }
  button.danger { background: transparent; color: var(--danger); border: 1px solid rgba(248,113,113,.35);
          padding: 5px 12px; border-radius: 8px; cursor: pointer; transition: all .15s; }
  button.danger:hover { background: rgba(248,113,113,.1); }
  /* 步骤指示器 */
  .steps { display: flex; margin: 20px 0 14px; }
  .step { flex: 1; text-align: center; position: relative; }
  .step::before { content: ''; position: absolute; top: 15px; left: -50%; width: 100%; height: 2px;
           background: rgba(125, 138, 165, 0.25); z-index: 0; }
  .step:first-child::before { display: none; }
  .dot { width: 32px; height: 32px; border-radius: 50%; background: rgba(10, 16, 32, .9);
         border: 1.5px solid rgba(125, 138, 165, 0.4); display: inline-flex; align-items: center;
         justify-content: center; font-size: 12.5px; color: var(--muted); position: relative; z-index: 1; transition: all .25s; }
  .step.active .dot { border-color: var(--cyan); color: var(--cyan); animation: pulse 1.3s infinite; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(34,211,238,.45); } 50% { box-shadow: 0 0 0 9px rgba(34,211,238,0); } }
  .step.done .dot { background: linear-gradient(135deg, #10b981, #0d9488); border-color: transparent; color: #fff; box-shadow: 0 0 12px rgba(52,211,153,.45); }
  .step.done::before { background: linear-gradient(90deg, rgba(52,211,153,.7), rgba(34,211,238,.7)); }
  .step .label { font-size: 12px; margin-top: 8px; color: var(--muted); letter-spacing: .4px; }
  .step.done .label, .step.active .label { color: #d9f7ff; }
  /* 终端风日志 */
  .log { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; color: #9fb0cc;
         line-height: 2; max-height: 130px; overflow-y: auto; background: rgba(2, 6, 18, 0.7);
         border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; margin: 12px 0; }
  .log::before { content: '> live log'; display: block; color: #3d4a66; font-size: 10.5px; letter-spacing: 2px; margin-bottom: 4px; }
  .log .ok { color: var(--ok); }
  /* 子 agent 实时输出面板 */
  .stage-panel { border: 1px solid var(--border); border-radius: 12px;
          background: rgba(2, 6, 18, 0.45); margin-bottom: 10px; overflow: hidden; }
  .stage-panel summary { cursor: pointer; padding: 11px 16px; font-size: 13px; color: #b9d4ff;
          letter-spacing: .4px; list-style: none; display: flex; align-items: center; gap: 8px; user-select: none; }
  .stage-panel summary::-webkit-details-marker { display: none; }
  .stage-panel summary::before { content: '▸'; color: var(--cyan); transition: transform .2s; }
  .stage-panel[open] summary::before { transform: rotate(90deg); }
  .stage-panel[open] summary { border-bottom: 1px solid var(--border); }
  .stage-body { padding: 12px 16px; max-height: 340px; overflow-y: auto; font-size: 12.8px;
          line-height: 1.75; color: #a9bce0; }
  .stage-live { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px;
          color: #7fd6e8; white-space: pre-wrap; word-break: break-word; }
  .hint.ok { color: var(--ok); }
  .hint.err { color: var(--danger); }
  /* 表格 */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 11px 12px; border-bottom: 1px solid var(--border-strong);
       color: var(--muted); font-weight: 600; letter-spacing: .5px; font-size: 12px; }
  td { padding: 11px 12px; border-bottom: 1px solid rgba(125, 138, 165, 0.12); vertical-align: top; }
  tr:hover td { background: rgba(34, 211, 238, 0.04); }
  a { color: var(--cyan); text-decoration: none; }
  a:hover { text-shadow: 0 0 10px rgba(34,211,238,.6); }
  /* Markdown 渲染 */
  .md { line-height: 1.8; font-size: 13.8px; word-break: break-word; }
  .md h1 { font-size: 21px; padding-bottom: 10px; margin: 24px 0 14px;
           border-bottom: 1px solid var(--border-strong);
           background: linear-gradient(90deg, #67e8f9, #a78bfa); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .md h2 { font-size: 17px; margin: 26px 0 12px; padding-left: 12px; color: #dbeafe;
           border-left: 3px solid var(--cyan); text-shadow: 0 0 16px rgba(34,211,238,.25); }
  .md h3 { font-size: 15px; margin: 20px 0 10px; color: #c7d9ff; }
  .md h4 { font-size: 13.8px; margin: 14px 0 8px; color: #b8c7ea; }
  .md p { margin: 8px 0; }
  .md ul, .md ol { padding-left: 26px; margin: 8px 0; }
  .md li { margin: 4px 0; }
  .md code { background: rgba(34, 211, 238, 0.09); border: 1px solid rgba(34,211,238,.15);
          border-radius: 5px; padding: 1.5px 7px; font-size: 12.3px; color: #9beaf7;
          font-family: ui-monospace, "SF Mono", Menlo, monospace; }
  .md pre { background: #05080f; border: 1px solid var(--border); color: #c9d8f0; padding: 15px 18px;
          border-radius: 12px; overflow-x: auto; font-size: 12.3px; box-shadow: inset 0 0 30px rgba(34,211,238,.04); }
  .md pre code { background: none; border: none; padding: 0; color: inherit; }
  .md table { margin: 12px 0; }
  .md th { background: rgba(34, 211, 238, 0.07); }
  .md blockquote { border-left: 3px solid var(--violet); background: rgba(139, 92, 246, 0.07);
          margin: 12px 0; padding: 10px 16px; border-radius: 0 10px 10px 0; color: #c4b5fd; }
  .md hr { border: none; border-top: 1px dashed var(--border-strong); margin: 20px 0; }
  .md strong { color: #fff; }
  .hint { color: var(--muted); font-size: 12px; }
  .status { color: var(--cyan); font-size: 13px; margin: 12px 0 0; white-space: pre-wrap; }
  .status.err { color: var(--danger); }
  .checkbox-list { max-height: 240px; overflow-y: auto; border: 1px solid var(--border);
          border-radius: 12px; padding: 12px 16px; background: rgba(2, 6, 18, 0.5); }
  .checkbox-list label { display: block; margin: 6px 0; cursor: pointer; transition: color .15s; }
  .checkbox-list label:hover { color: #d9f7ff; }
  .badge { display: inline-block; font-size: 11px; padding: 2.5px 10px; border-radius: 999px;
           background: rgba(34, 211, 238, 0.1); border: 1px solid rgba(34,211,238,.3); color: var(--cyan); margin-left: 6px; }
  .idea { border-left: 3px solid var(--cyan); padding-left: 16px; }
  .cite { display: inline-block; background: rgba(148, 180, 255, 0.06); border: 1px solid var(--border);
          border-radius: 8px; padding: 5px 12px; margin: 4px 8px 4px 0; font-size: 12px; color: #a9bce0; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(34,211,238,.2);
          border-top-color: var(--cyan); border-radius: 50%; animation: spin .8s linear infinite;
          vertical-align: -2px; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .empty { color: var(--muted); padding: 20px 0; text-align: center; }
  /* 原文图表画廊 */
  .fig-gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
  .fig-item { background: rgba(2, 6, 18, 0.5); border: 1px solid var(--border); border-radius: 12px;
          overflow: hidden; transition: all .2s; }
  .fig-item:hover { border-color: var(--border-strong); box-shadow: var(--glow); transform: translateY(-2px); }
  .fig-item img { width: 100%; display: block; background: #fff; }
  .fig-cap { padding: 7px 12px; font-size: 11.5px; color: var(--muted); letter-spacing: .5px; }
</style>
</head>
<body>
<header>
  <h1>📚 科研论文管理 Agent</h1>
  <p>论文解读 · 创新工坊 · Milvus 检索问答</p>
</header>
<nav>
  <button data-tab="interpret" class="active">📤 论文解读</button>
  <button data-tab="library">📚 论文库</button>
  <button data-tab="innovate">💡 创新工坊</button>
  <button data-tab="qa">🔍 问答</button>
</nav>
<main>

<!-- 论文解读 -->
<section class="panel active" id="panel-interpret">
  <div class="card">
    <h3>论文解读</h3>
    <div class="row">
      <input id="iq" placeholder="arXiv ID（如 1706.03762）· 论文库 ID（如 1）· 或自然语言描述">
      <button class="primary" id="btn-interpret" onclick="runInterpret()">开始解读</button>
    </div>
    <div class="steps" id="steps">
      <div class="step" data-kw="已定位|章节拆分|仓库已分析"><div class="dot">1</div><div class="label">准备资料</div></div>
      <div class="step" data-kw="背景调研"><div class="dot">2</div><div class="label">背景调研</div></div>
      <div class="step" data-kw="方法分析"><div class="dot">3</div><div class="label">方法分析</div></div>
      <div class="step" data-kw="实验分析"><div class="dot">4</div><div class="label">实验分析</div></div>
      <div class="step" data-kw="解读报告已生成"><div class="dot">5</div><div class="label">报告生成</div></div>
    </div>
    <div class="log" id="ilog" style="display:none"></div>
    <div id="stage-boxes" style="display:none">
      <details class="stage-panel" id="stage-background" open>
        <summary>🧭 背景调研 <span class="hint" data-hint="background">等待中…</span></summary>
        <div class="stage-body" data-stage="background"></div>
      </details>
      <details class="stage-panel" id="stage-method" open>
        <summary>⚙️ 方法分析 <span class="hint" data-hint="method">等待中…</span></summary>
        <div class="stage-body" data-stage="method"></div>
      </details>
      <details class="stage-panel" id="stage-experiment" open>
        <summary>🔬 实验分析 <span class="hint" data-hint="experiment">等待中…</span></summary>
        <div class="stage-body" data-stage="experiment"></div>
      </details>
    </div>
  </div>
  <div class="card" id="report-card" style="display:none">
    <h3>解读报告 <span class="hint" id="rpath"></span></h3>
    <div class="md" id="report"></div>
    <h3 style="margin-top:26px">📊 原文图表 <span class="hint" id="fig-hint"></span></h3>
    <div class="fig-gallery" id="fig-gallery"></div>
  </div>
</section>

<!-- 论文库 -->
<section class="panel" id="panel-library">
  <div class="card">
    <h3>上传 PDF 入库</h3>
    <div class="row">
      <input type="file" id="pdf-file" accept=".pdf" style="flex:1">
      <button class="ghost" onclick="uploadPdf()">入库</button>
    </div>
    <div class="status" id="ustatus"></div>
  </div>
  <div class="card">
    <h3>论文库 <span class="hint" id="pcount"></span></h3>
    <table><thead><tr><th>ID</th><th>标题</th><th>arXiv</th><th>分类</th><th>状态</th><th>报告</th><th></th></tr></thead>
    <tbody id="ptable"></tbody></table>
  </div>
  <div class="card" id="report-view-card" style="display:none">
    <h3 id="rview-title"></h3>
    <div class="md" id="report-view"></div>
    <h3 style="margin-top:26px">🧩 各阶段结果</h3>
    <div id="report-view-steps"></div>
    <h3 style="margin-top:26px">📊 原文图表</h3>
    <div class="fig-gallery" id="fig-gallery2"></div>
  </div>
</section>

<!-- 创新工坊 -->
<section class="panel" id="panel-innovate">
  <div class="card">
    <h3>选择论文（勾选多篇）</h3>
    <div class="checkbox-list" id="paper-picks"></div>
    <div class="row" style="margin-top:14px">
      <button class="primary" onclick="runExtract()">抽取创新点</button>
      <button class="ghost" onclick="runIdeas()">生成研究想法</button>
    </div>
    <div class="status" id="innov-status"></div>
  </div>
  <div class="card"><h3>历史研究想法</h3><div id="ideas-list"></div></div>
</section>

<!-- 问答 -->
<section class="panel" id="panel-qa">
  <div class="card">
    <h3>论文库问答</h3>
    <div class="row">
      <input id="q" placeholder="例如：Transformer 的核心贡献是什么？">
      <button class="primary" id="btn-qa" onclick="runQA()">提问</button>
    </div>
    <div class="log" id="qlog" style="display:none"></div>
    <div class="card" style="margin-top:14px; box-shadow:none; background:#fafbfe">
      <div class="md" id="answer"></div>
      <div id="citations" style="margin-top:12px"></div>
    </div>
  </div>
</section>

</main>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const j = r => r.json();

/* ---------- 标签切换 ---------- */
document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); $('panel-' + b.dataset.tab).classList.add('active');
  if (b.dataset.tab === 'library') loadPapers();
  if (b.dataset.tab === 'innovate') { loadPicks(); loadIdeas(); }
});

/* ---------- SSE 读取 ---------- */
async function readSSE(resp, onEvent, onResult, onDone) {
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const line = buf.slice(0, idx).trim(); buf = buf.slice(idx + 2);
      if (!line.startsWith('data: ')) continue;
      const d = JSON.parse(line.slice(6));
      if (d.event && onEvent) onEvent(d.event);
      else if (d.stage && onResult) onResult(d.stage, d.result);
      else if (d.done && onDone) onDone(d);
    }
  }
}

/* ---------- 解读 ---------- */
const STEPS = [...document.querySelectorAll('#steps .step')];
const STAGE_META = {
  background: {label: '背景调研', hint: $('stage-background').querySelector('[data-hint]')},
  method: {label: '方法分析', hint: $('stage-method').querySelector('[data-hint]')},
  experiment: {label: '实验分析', hint: $('stage-experiment').querySelector('[data-hint]')},
};
let stageRaw = {};

function stepFromEvent(ev) {
  for (let i = 0; i < STEPS.length; i++) {
    const kws = STEPS[i].dataset.kw.split('|');
    if (kws.some(k => k && ev.includes(k))) return i;
  }
  return -1;
}

function renderMathInto(el) {
  if (window.renderMathInElement) {
    try { renderMathInElement(el, {delimiters: [
      {left: '\\(', right: '\\)', display: false},
      {left: '\\[', right: '\\]', display: true},
      {left: '$$', right: '$$', display: true}], throwOnError: false}); } catch (e) {}
  }
}

function finalizeStage(stage, ok) {
  const body = document.querySelector('.stage-body[data-stage="' + stage + '"]');
  const hint = STAGE_META[stage].hint;
  if (body) {
    body.innerHTML = ok ? renderMarkdown(stageRaw[stage] || '（无输出）') : '<span style="color:#f87171">' + esc(stageRaw[stage] || '') + '</span>';
    renderMathInto(body);
  }
  hint.textContent = ok ? '✓ 完成' : '✗ 失败';
  hint.className = 'hint ' + (ok ? 'ok' : 'err');
  $('stage-' + stage).open = false;
}

async function loadFigures(paperId, container) {
  const figs = await fetch('/api/papers/' + paperId + '/figures').then(j);
  $('fig-hint').textContent = figs.length ? '（' + figs.length + ' 张，点击查看大图）' : '';
  container.innerHTML = figs.length ? figs.map(f =>
    `<div class="fig-item"><a href="${f.url}" target="_blank"><img src="${f.url}" loading="lazy" alt="图"></a>
     <div class="fig-cap">第 ${f.page} 页</div></div>`).join('')
    : '<div class="hint">未从 PDF 中提取到图片（纯文字论文或图表为矢量绘制）</div>';
}

async function runInterpret(q) {
  const query = (q || $('iq').value).trim();
  if (!query) return;
  const btn = $('btn-interpret'); btn.disabled = true;
  STEPS.forEach(s => s.classList.remove('done', 'active'));
  STEPS[0].classList.add('active');
  const log = $('ilog'); log.style.display = 'block'; log.innerHTML = '';
  $('stage-boxes').style.display = 'block';
  stageRaw = {background: '', method: '', experiment: ''};
  for (const stage of Object.keys(STAGE_META)) {
    document.querySelector('.stage-body[data-stage="' + stage + '"]').innerHTML = '<div class="hint">等待调度…</div>';
    STAGE_META[stage].hint.textContent = '等待中…';
    STAGE_META[stage].hint.className = 'hint';
    $('stage-' + stage).open = true;
  }
  $('report-card').style.display = 'block';
  $('report').innerHTML = '<div class="empty">正在准备资料…</div>';
  $('rpath').textContent = '';
  try {
    const resp = await fetch('/api/interpret/stream', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query}),
    });
    await readSSE(resp,
      ev => {
        log.innerHTML += '<span class="ok">✓</span> ' + esc(ev) + '<br>';
        log.scrollTop = log.scrollHeight;
        const i = stepFromEvent(ev);
        if (i >= 0) {
          STEPS[i].classList.add('done');
          STEPS[i].classList.remove('active');
          if (i + 1 < STEPS.length && !STEPS[i + 1].classList.contains('done')) STEPS[i + 1].classList.add('active');
        }
      },
      (stage, result) => {
        // 子 agent 完成：立即渲染该阶段的完整结果
        stageRaw[stage] = result || '';
        finalizeStage(stage, true);
      },
      d => {
        btn.disabled = false;
        STEPS.forEach(s => s.classList.remove('active'));
        if (d.error) {
          log.innerHTML += '<span style="color:#f87171">✗ ' + esc(d.error) + '</span>';
          $('report').innerHTML = '<div class="empty" style="color:#f87171">' + esc(d.error) + '</div>';
        } else {
          STEPS[STEPS.length - 1].classList.add('done');
          $('rpath').textContent = '（已保存 · 论文 #' + d.paper_id + '）';
          $('report').innerHTML = renderMarkdown(d.report || '');
          renderMathInto($('report'));
          loadFigures(d.paper_id, $('fig-gallery'));
          $('report-card').scrollIntoView({behavior: 'smooth'});
        }
      });
  } catch (e) {
    btn.disabled = false;
    $('report').innerHTML = '<div class="empty" style="color:#f87171">请求失败：' + esc(e) + '</div>';
  }
}

/* ---------- 论文库 ---------- */
async function loadPapers() {
  const ps = await fetch('/api/papers').then(j);
  $('pcount').textContent = '共 ' + ps.length + ' 篇';
  $('ptable').innerHTML = ps.map(p => `<tr>
    <td><b>${p.id}</b></td>
    <td>${esc(p.title)}<br><span class="hint">${esc(p.title_orig).slice(0, 60)}${p.title_orig.length > 60 ? '…' : ''}</span></td>
    <td>${esc(p.arxiv_id || '—')}</td>
    <td>${p.classification ? '<span class="badge">' + esc(p.classification) + '</span>' : '—'}</td>
    <td>${esc(p.status || '')}</td>
    <td>${p.has_report ? '<a href="#" onclick="viewReport(' + p.id + ')">查看</a>' : '—'}</td>
    <td><button class="danger" onclick="delPaper(${p.id})">删除</button></td></tr>`).join('') || '<tr><td colspan="7" class="empty">论文库为空，去「论文解读」上传 PDF 或输入 arXiv ID</td></tr>';
}

async function viewReport(id) {
  const r = await fetch('/api/papers/' + id + '/report').then(j);
  $('report-view-card').style.display = 'block';
  $('rview-title').textContent = '论文 #' + id + ' 解读报告';
  $('report-view').innerHTML = renderMarkdown(r.report || r.error || '');
  renderMathInto($('report-view'));
  $('report-view-steps').innerHTML = Object.entries(r.steps || {}).map(([label, content]) =>
    `<details class="stage-panel"><summary>${esc(label)}</summary><div class="stage-body">` +
    renderMarkdown(content) + `</div></details>`).join('') || '<div class="hint">该报告未保存阶段结果</div>';
  $('report-view-steps').querySelectorAll('.stage-body').forEach(renderMathInto);
  loadFigures(id, $('fig-gallery2'));
  $('report-view-card').scrollIntoView({behavior: 'smooth'});
}

async function delPaper(id) {
  if (!confirm('确认删除论文 #' + id + '？将同时删除 Milvus 索引与库内元数据。')) return;
  await fetch('/api/papers/' + id, {method: 'DELETE'});
  loadPapers();
}

async function uploadPdf() {
  const f = $('pdf-file').files[0];
  if (!f) return;
  const fd = new FormData(); fd.append('file', f);
  $('ustatus').textContent = '⏳ 入库中：解析 PDF → 分块 → 建立 Milvus 索引…';
  const r = await fetch('/api/ingest/pdf', {method: 'POST', body: fd}).then(j);
  $('ustatus').textContent = (r.events || []).join('\n') || JSON.stringify(r);
  $('ustatus').className = 'status' + (r.error ? ' err' : '');
  if (r.paper_id && !r.error) runInterpret('paper:' + r.paper_id);
}

/* ---------- 创新工坊 ---------- */
async function loadPicks() {
  const ps = await fetch('/api/papers').then(j);
  $('paper-picks').innerHTML = ps.length ? ps.map(p =>
    `<label><input type="checkbox" value="${p.id}"> #${p.id} · ${esc(p.title)}</label>`).join('')
    : '<div class="empty">论文库为空</div>';
}
const pickedIds = () => [...document.querySelectorAll('#paper-picks input:checked')].map(i => +i.value);

async function runExtract() {
  const ids = pickedIds(); if (!ids.length) return;
  $('innov-status').textContent = '<span class="spinner"></span>抽取创新点中…（通常需要几分钟）';
  const r = await fetch('/api/innovations/extract', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({paper_ids: ids})}).then(j);
  $('innov-status').innerHTML = (r.events || []).map(esc).join('<br>') || esc(r.error || '完成');
  loadPicks();
}

async function runIdeas() {
  const ids = pickedIds(); if (!ids.length) return;
  $('innov-status').innerHTML = '<span class="spinner"></span>组合创新点生成想法中…';
  const r = await fetch('/api/ideas', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({paper_ids: ids})}).then(j);
  $('innov-status').innerHTML = (r.events || []).map(esc).join('<br>') || esc(r.error || '完成');
  loadIdeas();
}

async function loadIdeas() {
  const ideas = await fetch('/api/ideas').then(j);
  $('ideas-list').innerHTML = ideas.length ? ideas.map(i => `<div class="card idea">
    <b>💡 ${esc(i.title)}</b>
    <p>${esc(i.hypothesis)}</p>
    <p class="hint">组合方式：${esc(i.combination)}</p>
    <p class="hint">可行性：${esc(i.feasibility || '—')}</p>
    <p class="hint">风险：${esc((i.risks || []).join('；'))}</p>
    <p class="hint">实验建议：${esc((i.experiments || []).join('；'))}</p>
    </div>`).join('') : '<div class="empty">还没有生成过研究想法</div>';
}

/* ---------- 问答 ---------- */
async function runQA() {
  const q = $('q').value.trim(); if (!q) return;
  const btn = $('btn-qa'); btn.disabled = true;
  const log = $('qlog'); log.style.display = 'block'; log.innerHTML = '';
  $('answer').innerHTML = '<div class="empty"><span class="spinner"></span>检索论文库…</div>';
  $('citations').innerHTML = '';
  try {
    const resp = await fetch('/api/qa/stream', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q}),
    });
    await readSSE(resp,
      ev => { log.innerHTML += '<span class="ok">✓</span> ' + esc(ev) + '<br>'; },
      () => {},
      d => {
        btn.disabled = false;
        if (d.error) $('answer').innerHTML = '<div class="empty" style="color:#f87171">' + esc(d.error) + '</div>';
        else {
          $('answer').innerHTML = renderMarkdown(d.answer || '');
          renderMathInto($('answer'));
          $('citations').innerHTML = (d.citations || []).map(c =>
            `<span class="cite">📎 《${esc(c.title)}》${c.arxiv_id ? ' arXiv:' + esc(c.arxiv_id) : ''}</span>`).join('');
        }
      });
  } catch (e) {
    btn.disabled = false;
    $('answer').innerHTML = '<div class="empty" style="color:#dc2626">请求失败：' + esc(e) + '</div>';
  }
}

/* ---------- 迷你 Markdown 渲染 ---------- */
function renderMarkdown(src) {
  let s = esc(src);
  // 代码块
  s = s.replace(/```(\w*)\n?([\s\S]*?)```/g, (m, lang, code) => '<pre><code>' + code.replace(/\n$/, '') + '</code></pre>');
  // 表格
  s = s.replace(/((?:^\|.+\|\n)+)(?=\n|$)/gm, block => {
    const rows = block.trim().split('\n').filter(r => /\|/.test(r));
    const toCells = r => r.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
    const isSep = r => /^[\s:|-]+$/.test(r);
    const head = toCells(rows[0]).map(c => '<th>' + c + '</th>').join('');
    const body = rows.slice(1).filter(r => !isSep(r))
      .map(r => '<tr>' + toCells(r).map(c => '<td>' + c + '</td>').join('') + '</tr>').join('');
    return '<table><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table>';
  });
  // 标题
  s = s.replace(/^#### (.*)$/gm, '<h4>$1</h4>');
  s = s.replace(/^### (.*)$/gm, '<h3>$1</h3>');
  s = s.replace(/^## (.*)$/gm, '<h2>$1</h2>');
  s = s.replace(/^# (.*)$/gm, '<h1>$1</h1>');
  // 列表（连续行）
  s = s.replace(/(^[-*] .+(?:\n|$))+/gm, m => '<ul>' + m.trim().split('\n').map(x => '<li>' + x.replace(/^[-*] /, '') + '</li>').join('') + '</ul>');
  s = s.replace(/(^\d+\. .+(?:\n|$))+/gm, m => '<ol>' + m.trim().split('\n').map(x => '<li>' + x.replace(/^\d+\. /, '') + '</li>').join('') + '</ol>');
  // 引用 / 分割线
  s = s.replace(/(^&gt; .+(?:\n|$))+/gm, m => '<blockquote>' + m.trim().split('\n').map(x => x.replace(/^&gt; /, '')).join('<br>') + '</blockquote>');
  s = s.replace(/^\s*(?:-{3,}|\*{3,})\s*$/gm, '<hr>');
  // 行内
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  s = s.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  // 段落
  s = s.replace(/\n{2,}/g, '\n');
  s = s.split('\n').map(line =>
    /^<(h\d|ul|ol|table|pre|blockquote|hr)/.test(line) ? line : '<p>' + line + '</p>'
  ).join('\n');
  return s;
}

loadPapers(); loadPicks(); loadIdeas();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8501)

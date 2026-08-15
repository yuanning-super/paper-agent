"""入库管线：arXiv 元数据抓取、PDF 下载/提取、分块、嵌入、LLM 元数据增强。"""

from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .config import load_settings
from .db import (
    Chunk,
    get_meta_int,
    get_paper,
    insert_paper,
    replace_chunks,
    set_meta,
    update_paper,
)
from .embed import get_embedder
from .utils import detect_heading, split_sentences_zh

logger = logging.getLogger("paper_agent")

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
ARXIV_ID_RE = re.compile(r"(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
GITHUB_RE = re.compile(r"github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
CACHE_TTL = 7 * 24 * 3600  # 7 天


@dataclass
class IngestResult:
    paper_id: int | None = None
    is_new: bool = False
    title: str = ""
    status: str = "metadata"
    is_scanned: bool = False
    chunk_count: int = 0
    error: str | None = None
    events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "is_new": self.is_new,
            "title": self.title,
            "status": self.status,
            "is_scanned": self.is_scanned,
            "chunk_count": self.chunk_count,
            "error": self.error,
        }


def normalize_arxiv_id(raw: str) -> str:
    """'arXiv:1706.03762v2' / '1706.03762' → '1706.03762'。非法输入返回原字符串。"""
    m = ARXIV_ID_RE.search(raw.strip())
    return m.group(1) if m else raw.strip()


# ---------- arXiv 元数据 ----------

def fetch_arxiv_metadata(arxiv_id: str, use_cache: bool = True) -> dict:
    """从 export.arxiv.org 获取元数据（stdlib XML 解析，磁盘缓存 7 天）。失败抛异常。"""
    arxiv_id = normalize_arxiv_id(arxiv_id)
    cache_path = load_settings().cache_dir / f"{arxiv_id}.xml"
    if use_cache and cache_path.exists() and time.time() - cache_path.stat().st_mtime < CACHE_TTL:
        content = cache_path.read_text(encoding="utf-8")
    else:
        resp = requests.get(
            "https://export.arxiv.org/api/query",
            params={"id_list": arxiv_id, "max_results": 1},
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.text
        cache_path.write_text(content, encoding="utf-8")

    root = ET.fromstring(content)
    entry = root.find("atom:entry", ATOM_NS)
    if entry is None:
        raise ValueError(f"arXiv 未找到论文 {arxiv_id}")

    def _text(tag: str) -> str:
        el = entry.find(f"atom:{tag}", ATOM_NS)
        return (el.text or "").strip() if el is not None else ""

    authors = [
        a.find("atom:name", ATOM_NS).text or ""
        for a in entry.findall("atom:author", ATOM_NS)
        if a.find("atom:name", ATOM_NS) is not None
    ]
    primary = entry.find("arxiv:primary_category", ATOM_NS)
    categories = [c.get("term", "") for c in entry.findall("atom:category", ATOM_NS)]
    doi_el = entry.find("arxiv:doi", ATOM_NS)

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    for link in entry.findall("atom:link", ATOM_NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", pdf_url)

    # comment 字段常含 GitHub 链接线索
    comment = entry.find("arxiv:comment", ATOM_NS)
    comment_text = comment.text or "" if comment is not None else ""
    gh = GITHUB_RE.search(comment_text + " " + _text("summary"))

    return {
        "arxiv_id": arxiv_id,
        "title": re.sub(r"\s+", " ", _text("title")),
        "authors": [a.strip() for a in authors if a.strip()],
        "abstract": re.sub(r"\s+", " ", _text("summary")),
        "published": (_text("published") or "")[:10],
        "categories": [c for c in categories if c],
        "primary_category": primary.get("term") if primary is not None else None,
        "doi": doi_el.text if doi_el is not None else None,
        "pdf_url": pdf_url,
        "comment": comment_text,
        "github_hint": gh.group(0) if gh else None,
    }


# ---------- PDF ----------

def download_pdf(arxiv_id: str, pdf_url: str) -> Path:
    settings = load_settings()
    dest = settings.pdfs_dir / f"{arxiv_id}.pdf"
    if dest.exists():
        return dest
    resp = requests.get(pdf_url, stream=True, timeout=180)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return dest


def extract_text(pdf_path: Path) -> tuple[str, bool]:
    """pymupdf 提取全文，返回 (text, is_scanned)。"""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    text = "\n\n".join(pages)
    return text, len(text.strip()) < 200


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """按字符窗口分块，并带章节标题启发式标注。"""
    settings = load_settings()
    size = size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    pieces = split_sentences_zh(text, size, overlap)
    chunks: list[Chunk] = []
    current_heading: str | None = None
    for i, piece in enumerate(pieces):
        first_line = piece.splitlines()[0].strip() if piece else ""
        heading = detect_heading(first_line)
        if heading:
            current_heading = heading
        chunks.append(Chunk(paper_id=0, seq=i, content=piece, heading=current_heading))
    return chunks


# ---------- LLM 元数据增强 ----------

def enrich_metadata(paper_id: int) -> bool:
    """LLM 生成中文标题/关键词/分类。失败仅告警，不阻塞主流程。"""
    import os

    if os.environ.get("PAPER_AGENT_SKIP_LLM") == "1":
        return False

    from . import llm  # 延迟导入避免循环依赖
    from .schemas import CLASSIFICATION_OPTIONS

    paper = get_paper(paper_id)
    if not paper or not paper.get("abstract"):
        return False
    prompt = (
        f"论文标题：{paper['title']}\n摘要：{paper['abstract'][:1500]}\n"
        f"请输出 JSON：{{\"title_zh\": \"标题的中文翻译\", "
        f"\"keywords\": [\"5-8个中文关键词\"], "
        f"\"classification\": \"研究方向分类\"}}\n"
        f"classification 必须从以下候选中选择一个：{' / '.join(CLASSIFICATION_OPTIONS)}"
    )

    def _validate(data) -> str | None:
        if not isinstance(data, dict):
            return "输出必须是 JSON 对象"
        if not data.get("title_zh"):
            return "title_zh 缺失"
        if not isinstance(data.get("keywords"), list) or not 3 <= len(data["keywords"]) <= 10:
            return "keywords 必须是 3-10 个词的数组"
        if not data.get("classification"):
            return "classification 缺失"
        return None

    try:
        data, _ = llm.complete_json(prompt, validator=_validate)
        if not data:
            logger.warning("论文 %d 元数据增强失败，保留原始元数据", paper_id)
            return False
        if data.get("classification") not in CLASSIFICATION_OPTIONS:
            data["classification"] = "其他"
        update_paper(
            paper_id,
            title_zh=data.get("title_zh"),
            keywords=json.dumps(data.get("keywords", []), ensure_ascii=False),
            classification=data.get("classification"),
        )
        return True
    except Exception as e:  # noqa: BLE001 —— 增强失败绝不影响入库主流程
        logger.warning("论文 %d 元数据增强失败（%s），保留原始元数据", paper_id, e)
        return False


# ---------- 完整入库 ----------

def ingest_arxiv(arxiv_id: str, checkpointer=None) -> IngestResult:
    """arXiv 论文全流程入库：委托给 LangGraph 入库工作流（见 graphs/ingest_graph.py）。"""
    from .graphs.ingest_graph import run_ingest  # 延迟导入避免循环依赖

    return run_ingest(arxiv_id, checkpointer=checkpointer)


def ingest_pdf_file(path: str | Path) -> IngestResult:
    """上传 PDF 入库：标题从文本首行启发式提取，无 arXiv 元数据。"""
    result = IngestResult()
    path = Path(path)

    text, is_scanned = extract_text(path)
    result.is_scanned = is_scanned
    if is_scanned:
        result.error = "扫描版 PDF：未提取到文本内容。建议上传文本版 PDF 或启用 OCR。"
        result.events.append(result.error)
        return result

    # 标题启发式：取前 5 个非空行中最长的一行
    lines = [ln.strip() for ln in text.splitlines()[:20] if ln.strip() and len(ln.strip()) > 10]
    title = max(lines, key=len) if lines else path.stem
    title = re.sub(r"\s+", " ", title)[:200]

    # 简单去重：标题完全相同视为重复
    existing = _find_paper_by_title(title)
    if existing:
        result.paper_id = existing["id"]
        result.is_new = False
        result.title = existing["title"]
        result.status = existing["status"]
        result.events.append(f"库中已存在标题相同的论文（#{existing['id']}），跳过入库")
        return result

    paper_id, _ = insert_paper({"title": title, "source": "upload"})
    result.paper_id = paper_id
    result.is_new = True
    result.title = title
    result.events.append(f"论文已登记（#{paper_id}）：{title}")

    from shutil import copy2

    settings = load_settings()
    dest = settings.pdfs_dir / f"upload_{paper_id}.pdf"
    copy2(path, dest)
    update_paper(paper_id, pdf_path=str(dest), status="text")

    chunks = chunk_text(text)
    result.chunk_count = len(chunks)
    replace_chunks(paper_id, chunks)
    update_paper(paper_id, status="chunked")
    result.events.append(f"分块完成：{len(chunks)} 块")

    embedder = get_embedder()
    if embedder.available:
        vecs = embedder.embed_many([c.content for c in chunks])
        if vecs is not None:
            for c, v in zip(chunks, vecs):
                c.embedding = v.tobytes()
            replace_chunks(paper_id, chunks)
            update_paper(paper_id, status="embedded")
            result.events.append("向量嵌入完成")
            # 写入 Milvus 向量索引
            from .rag.pipeline import index_embedded_chunks

            r = index_embedded_chunks(paper_id, chunks, vecs)
            if r.get("ok"):
                result.events.append(f"已写入 Milvus 索引（{r['indexed']} 块）")
            else:
                result.events.append(f"Milvus 索引失败（{r.get('error')}），检索降级纯 BM25")

    # 用提取文本生成摘要用于增强
    update_paper(paper_id, abstract=text[:2000])
    if enrich_metadata(paper_id):
        result.events.append("元数据增强完成（中文标题/关键词/分类）")
    update_paper(paper_id, status="enriched")
    set_meta("index_version", str(get_meta_int("index_version") + 1))
    result.status = "enriched"
    result.events.append("入库完成 ✓")
    return result


def _find_paper_by_title(title: str) -> dict | None:
    from .db import list_papers

    normalized = re.sub(r"\s+", " ", title).strip().lower()
    for p in list_papers():
        if re.sub(r"\s+", " ", p["title"]).strip().lower() == normalized:
            return p
    return None

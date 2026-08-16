"""Agent 工具集（LangChain @tool）：arXiv、GitHub 仓库分析、论文库查询。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from pathlib import Path

import requests
from langchain_core.tools import tool

from .config import load_settings
from .db import get_chunks, get_paper, update_paper
from .db import get_innovations as db_get_innovations
from .ingestion import (
    extract_text,
    fetch_arxiv_metadata,
    ingest_arxiv,
    normalize_arxiv_id,
)
from .retrieval import search as hybrid_search
from .utils import truncate

logger = logging.getLogger("paper_agent")

GITHUB_REPO_RE = re.compile(r"(?:https?://)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")


def _limit(data: dict) -> str:
    text = json.dumps(data, ensure_ascii=False)
    if len(text) <= load_settings().tool_result_limit:
        return text
    # 超限时保留头部（保持 JSON 前半段可读），并注明截断
    return truncate(text, load_settings().tool_result_limit, mode="head") + "\n（JSON 输出过长已截断，可缩小查询范围重试）"


def _error(msg: str) -> str:
    return _limit({"ok": False, "error": msg})


# ---------- arXiv ----------

@tool
def arxiv_fetch_metadata(arxiv_id: str) -> str:
    """获取 arXiv 论文元数据（标题/作者/摘要/分类/PDF 链接等）。
    当需要了解某篇论文的基本信息、或其 comment 字段中的 GitHub 线索时调用。"""
    try:
        meta = fetch_arxiv_metadata(normalize_arxiv_id(arxiv_id))
        meta["abstract"] = truncate(meta.get("abstract", ""), 500)
        return _limit({"ok": True, **meta})
    except Exception as e:  # noqa: BLE001
        return _error(f"arXiv 元数据获取失败：{e}")


@tool
def ingest_arxiv_paper(arxiv_id: str) -> str:
    """把 arXiv 论文完整入库：下载 PDF、提取全文、分块、向量化、元数据增强。
    论文已在库中时返回已有记录（is_new=false），不会重复入库。"""
    result = ingest_arxiv(arxiv_id)
    return _limit({"ok": result.error is None, **result.to_dict(), "events": result.events[-3:]})


# ---------- 论文库 ----------

@tool
def search_library(query: str, top_k: int = 5) -> str:
    """在论文库中检索（Milvus BM25 倒排；启用 embedding 时为混合检索）。返回与查询最相关的论文片段。
    当用户询问论文库内容、或需要引用具体论文细节时调用。"""
    try:
        hits = hybrid_search(query, top_k=top_k)
        return _limit({"ok": True, "count": len(hits), "hits": [h.to_dict() for h in hits]})
    except Exception as e:  # noqa: BLE001
        return _error(f"检索失败：{e}")


@tool
def get_paper_summary(paper_id: int) -> str:
    """获取库中某篇论文的元数据摘要（标题/作者/摘要/入库状态/分块数）。
    需要引用某篇论文的背景信息时调用。"""
    paper = get_paper(paper_id)
    if not paper:
        return _error(f"论文 #{paper_id} 不存在")
    return _limit(
        {
            "ok": True,
            "paper_id": paper["id"],
            "arxiv_id": paper.get("arxiv_id"),
            "title": paper["title"],
            "title_zh": paper.get("title_zh"),
            "authors": paper.get("authors", [])[:10],
            "abstract": truncate(paper.get("abstract") or "", 800),
            "status": paper.get("status"),
            "github_url": paper.get("github_url"),
            "chunk_count": len(get_chunks(paper_id)),
        }
    )


@tool
def get_innovations(paper_id: int) -> str:
    """获取某篇论文已抽取的创新点列表。生成新研究想法前调用，收集组合素材。"""
    items = db_get_innovations(paper_id)
    return _limit({"ok": True, "count": len(items), "innovations": items})


# ---------- GitHub ----------

@tool
def find_github_url(paper_id: int) -> str:
    """查找论文的官方 GitHub 代码仓库。依次扫描：已记录链接 → arXiv comment → PDF 全文 → GitHub 搜索 API。
    命中后回填论文记录。找不到时 source 为 none，不要编造。"""
    paper = get_paper(paper_id)
    if not paper:
        return _error(f"论文 #{paper_id} 不存在")

    if paper.get("github_url"):
        return _limit({"ok": True, "url": paper["github_url"], "source": "recorded"})

    # ① arXiv comment（元数据入库时已提取过 github_hint 并回填，此处兜底再扫摘要）
    abstract_hint = GITHUB_REPO_RE.search(paper.get("abstract") or "")
    if abstract_hint:
        url = "https://github.com/" + abstract_hint.group(1) + "/" + abstract_hint.group(2)
        update_paper(paper_id, github_url=url)
        return _limit({"ok": True, "url": url, "source": "abstract"})

    # ② PDF 全文
    if paper.get("pdf_path") and Path(paper["pdf_path"]).exists():
        try:
            text, _ = extract_text(Path(paper["pdf_path"]))
            for m in GITHUB_REPO_RE.finditer(text):
                url = "https://github.com/" + m.group(1) + "/" + m.group(2)
                update_paper(paper_id, github_url=url)
                return _limit({"ok": True, "url": url, "source": "pdf_text"})
        except Exception as e:  # noqa: BLE001
            logger.warning("扫描 PDF 文本失败：%s", e)

    # ③ GitHub 搜索 API（未认证限流 10 次/分，403 静默降级）
    try:
        query = re.sub(r"[^\w\s-]", " ", paper["title"])[:80]
        headers = {}
        if os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "per_page": 3},
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            if items:
                url = items[0]["html_url"]
                update_paper(paper_id, github_url=url)
                return _limit({"ok": True, "url": url, "source": "github_api"})
    except Exception as e:  # noqa: BLE001
        logger.warning("GitHub 搜索失败：%s", e)

    return _limit({"ok": True, "url": None, "source": "none"})


@tool
def analyze_github_repo(url: str) -> str:
    """浅克隆 GitHub 仓库到论文工作区并分析：README 要点、目录结构、依赖清单、核心代码文件摘要。
    返回的结构化结果将作为解读报告"代码仓库分析"章节的素材。"""
    url = url.strip().rstrip("/")
    m = GITHUB_REPO_RE.search(url)
    if not m:
        return _error(f"不是有效的 GitHub 仓库地址：{url}")
    clean_url = f"https://github.com/{m.group(1)}/{m.group(2)}"

    try:
        repo = Path(_get_repo_clone(clean_url))
        readme = _find_readme(repo)
        tree = _dir_tree(repo, max_entries=40)
        deps = _extract_deps(repo)
        core = _core_files(repo)
        stats = _file_stats(repo)
    except Exception as e:  # noqa: BLE001
        return _error(f"仓库分析失败：{e}")

    return _limit(
        {
            "ok": True,
            "url": clean_url,
            "readme_excerpt": readme[:2000] if readme else "",
            "tree": tree,
            "deps": deps,
            "core_files": core,
            "stats": stats,
        }
    )


# ---------- 解读专用：章节与仓库文件按需读取 ----------

_section_cache: dict[int, list[dict]] = {}


def set_section_cache(paper_id: int, sections: list[dict]) -> None:
    """解读流水线在章节拆分后写入，供 read_section 按需读取（进程内缓存）。"""
    _section_cache[paper_id] = sections


def get_section_cache(paper_id: int) -> list[dict]:
    return _section_cache.get(paper_id, [])


@tool
def read_section(paper_id: int, index: int) -> str:
    """按编号读取论文章节的全文（编号从 0 开始，与章节目录一致）。
    先看章节目录，需要哪节细节时再按需取，避免一次性读入全文。"""
    sections = _section_cache.get(paper_id)
    if not sections:
        return _error(f"论文 #{paper_id} 的章节尚未加载")
    if not 0 <= index < len(sections):
        return _error(f"章节编号 {index} 越界（共 {len(sections)} 节）")
    sec = sections[index]
    content = sec["content"] if len(sec["content"]) <= 12_000 else truncate(sec["content"], 12_000, mode="head")
    return json.dumps(
        {"ok": True, "index": index, "title": sec["title"], "content": content},
        ensure_ascii=False,
    )


_repo_cache: dict[str, str] = {}  # 仓库 url → 本地克隆目录（进程内复用）


def _repo_dir(url: str) -> Path:
    """论文代码统一放在论文自己的工作区（workspaces/{paper_id}/repo）；
    未关联到论文的仓库退回共享克隆目录。"""
    from .db import find_paper_by_github_url

    settings = load_settings()
    paper = find_paper_by_github_url(url)
    if paper:
        return settings.workspaces_dir / str(paper["id"]) / "repo"
    return settings.clones_dir / "repos" / hashlib.sha1(url.encode()).hexdigest()[:12]


def _get_repo_clone(url: str) -> str:
    if url in _repo_cache and Path(_repo_cache[url]).exists():
        return _repo_cache[url]
    dest = _repo_dir(url)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", url, str(dest)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"克隆失败：{(proc.stderr or '').strip()[-300:]}")
    _repo_cache[url] = str(dest)
    return str(dest)


@tool
def read_repo_file(url: str, path: str, start_line: int = 1, max_lines: int = 200) -> str:
    """读取 GitHub 仓库中的指定文件（相对路径，如 csrc/flash_attn/fwd.cu）。
    可用 start_line/max_lines 指定行范围，默认从头读 200 行。
    先用 grep_repo 定位代码位置，再用本工具精读对应文件。"""
    try:
        repo = Path(_get_repo_clone(url))
        target = (repo / path).resolve()
        if not target.is_file() or not target.is_relative_to(repo.resolve()):
            return _error(f"文件不存在或路径非法：{path}")
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        start = max(1, start_line) - 1
        text = "\n".join(lines[start : start + max_lines])
        return _limit({"ok": True, "path": path, "start_line": start + 1, "content": text})
    except Exception as e:  # noqa: BLE001
        return _error(f"读取失败：{e}")


@tool
def grep_repo(url: str, pattern: str, ignore_case: bool = True, max_matches: int = 30) -> str:
    """在 GitHub 仓库代码中搜索（ripgrep 正则），返回「路径:行号 匹配行」列表。
    定位类/函数定义、关键超参、调用位置时调用；命中后用 read_repo_file 精读对应文件。"""
    try:
        repo = Path(_get_repo_clone(url))
        cmd = ["rg", "--line-number", "--no-heading", "--no-messages", "-m", str(max_matches)]
        if ignore_case:
            cmd.append("--ignore-case")
        cmd += ["--", pattern, str(repo)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except FileNotFoundError:
        return _error("ripgrep（rg）未安装，请先安装：brew install ripgrep")
    except subprocess.TimeoutExpired:
        return _error("搜索超时（120 秒），可换更精确的 pattern 重试")
    except Exception as e:  # noqa: BLE001
        return _error(f"搜索失败：{e}")
    if proc.returncode == 2:
        return _error(f"搜索失败：{(proc.stderr or '').strip()[-300:]}")
    matches = [ln.strip()[:200] for ln in proc.stdout.splitlines() if ln.strip()]
    matches = matches[:max_matches]  # rg 的 -m 是每文件上限，这里再限总量
    return _limit(
        {
            "ok": True,
            "matches": matches,
            "truncated": len(matches) >= max_matches,
        }
    )


@tool
def list_repo_dir(url: str, path: str = "") -> str:
    """列出 GitHub 仓库中某目录的内容（path 为相对路径，留空列根目录），目录以 / 结尾。
    浏览代码结构、定位实现文件位置时调用。"""
    try:
        repo = Path(_get_repo_clone(url))
        target = (repo / path).resolve() if path else repo.resolve()
        if not target.is_dir() or not target.is_relative_to(repo.resolve()):
            return _error(f"目录不存在或路径非法：{path}")
        entries = []
        for p in sorted(target.iterdir()):
            if p.name.startswith("."):
                continue
            rel = p.relative_to(repo)
            entries.append(f"{rel}/" if p.is_dir() else str(rel))
        return _limit({"ok": True, "path": path or "/", "entries": entries})
    except Exception as e:  # noqa: BLE001
        return _error(f"读取失败：{e}")


# ---------- GitHub 仓库整体分析 ----------

def _find_readme(repo: Path) -> str:
    for name in ("README.md", "readme.md", "README.rst", "README"):
        p = repo / name
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    return ""


def _dir_tree(repo: Path, max_entries: int = 40) -> list[str]:
    entries: list[str] = []
    for path in sorted(repo.rglob("*")):
        rel = path.relative_to(repo)
        if any(part.startswith(".") and part != "." for part in rel.parts):
            continue  # 跳过 .git 等隐藏目录
        if rel.name.startswith("."):
            continue
        depth = len(rel.parts)
        if depth > 2 and not (depth == 3 and path.is_file()):
            continue  # 只展示两层目录 + 第二层内的文件
        entries.append(("  " * (depth - 1)) + rel.name + ("/" if path.is_dir() else ""))
        if len(entries) >= max_entries:
            entries.append("…（目录项过多已省略）")
            break
    return entries


def _extract_deps(repo: Path) -> dict:
    out = {}
    for name in ("requirements.txt", "pyproject.toml", "setup.py", "environment.yml"):
        p = repo / name
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="ignore")
            out[name] = text[:600]
    return out


def _core_files(repo: Path, limit: int = 3) -> list[dict]:
    """启发式找核心代码文件：文件名含 model/main/train 的 .py，或第一层的同名模块。"""
    candidates: list[Path] = []
    for p in repo.rglob("*.py"):
        if any(part.startswith(".") for part in p.relative_to(repo).parts):
            continue
        if p.stat().st_size > 300_000:
            continue
        lower = p.stem.lower()
        if any(k in lower for k in ("model", "main", "train", "network")):
            candidates.append(p)
    candidates = candidates[:limit]
    out = []
    for p in candidates:
        text = p.read_text(encoding="utf-8", errors="ignore")
        head = "\n".join(text.splitlines()[:120])
        out.append({"path": str(p.relative_to(repo)), "head": truncate(head, 2000)})
    return out


def _file_stats(repo: Path) -> dict:
    from collections import Counter

    counter = Counter()
    for p in repo.rglob("*"):
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(repo).parts):
            counter[p.suffix or "(no-ext)"] += 1
    return {"total_files": sum(counter.values()), "by_extension": dict(counter.most_common(10))}


# ---------- 联网搜索（背景调研） ----------

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """联网搜索（DuckDuckGo），返回标题/链接/摘要。
    调研论文之外的外部信息（数据集、baseline 方法背景）时调用。"""
    from duckduckgo_search import DDGS  # 延迟导入：首次联网搜索才加载

    try:
        with DDGS() as d:
            results = list(d.text(keywords=query, max_results=max_results))
    except Exception as e:  # noqa: BLE001
        return _error(f"网络搜索失败：{e}")
    items = [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")[:300]}
        for r in results
    ]
    return _limit({"ok": True, "count": len(items), "results": items})


# ---------- 汇总 ----------

TOOL_LIST = [
    arxiv_fetch_metadata,
    ingest_arxiv_paper,
    search_library,
    web_search,
    find_github_url,
    analyze_github_repo,
    grep_repo,
    list_repo_dir,
    read_repo_file,
    read_section,
    get_innovations,
    get_paper_summary,
]

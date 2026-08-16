"""SQLite 持久化：论文、分块、创新点、研究想法、问答日志。"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from .config import load_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  arxiv_id    TEXT UNIQUE,
  doi         TEXT,
  title       TEXT NOT NULL,
  title_zh    TEXT,
  authors     TEXT NOT NULL DEFAULT '[]',
  abstract    TEXT,
  categories  TEXT DEFAULT '[]',
  keywords    TEXT DEFAULT '[]',
  classification TEXT,
  published   TEXT,
  pdf_path    TEXT,
  github_url  TEXT,
  status      TEXT NOT NULL DEFAULT 'metadata',
  report_path TEXT,
  source      TEXT DEFAULT 'arxiv',
  created_at  TEXT DEFAULT (datetime('now')),
  updated_at  TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_papers_class ON papers(classification);

CREATE TABLE IF NOT EXISTS chunks (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id  INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  seq       INTEGER NOT NULL,
  heading   TEXT,
  content   TEXT NOT NULL,
  embedding BLOB,
  UNIQUE(paper_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);

CREATE TABLE IF NOT EXISTS innovations (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id      INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  kind          TEXT NOT NULL,
  title         TEXT NOT NULL,
  description   TEXT NOT NULL,
  novelty       TEXT,
  source_chunks TEXT DEFAULT '[]',
  raw_json      TEXT,
  created_at    TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_innov_paper ON innovations(paper_id);

CREATE TABLE IF NOT EXISTS ideas (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  title              TEXT NOT NULL,
  hypothesis         TEXT NOT NULL,
  combination        TEXT NOT NULL,
  source_innovations TEXT DEFAULT '[]',
  source_papers      TEXT DEFAULT '[]',
  feasibility        TEXT,
  risks              TEXT DEFAULT '[]',
  experiments        TEXT DEFAULT '[]',
  raw_json           TEXT,
  created_at         TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS qa_log (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  query      TEXT NOT NULL,
  answer     TEXT NOT NULL,
  citations  TEXT DEFAULT '[]',
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);
"""


def get_conn() -> sqlite3.Connection:
    """每次调用新建连接（WAL 模式，适合 Streamlit 多线程）。"""
    conn = sqlite3.connect(load_settings().db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# ---------- papers ----------

def insert_paper(meta: dict) -> tuple[int, bool]:
    """插入论文元数据；按 arxiv_id 去重，返回 (paper_id, is_new)。"""
    arxiv_id = meta.get("arxiv_id")
    with get_conn() as conn:
        if arxiv_id:
            row = conn.execute(
                "SELECT id FROM papers WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
            if row:
                return int(row["id"]), False
        cur = conn.execute(
            """INSERT INTO papers (arxiv_id, doi, title, title_zh, authors, abstract,
                categories, keywords, classification, published, source)
               VALUES (:arxiv_id, :doi, :title, :title_zh, :authors, :abstract,
                :categories, :keywords, :classification, :published, :source)""",
            {
                "arxiv_id": arxiv_id,
                "doi": meta.get("doi"),
                "title": meta.get("title", ""),
                "title_zh": meta.get("title_zh"),
                "authors": json.dumps(meta.get("authors", []), ensure_ascii=False),
                "abstract": meta.get("abstract"),
                "categories": json.dumps(meta.get("categories", []), ensure_ascii=False),
                "keywords": json.dumps(meta.get("keywords", []), ensure_ascii=False),
                "classification": meta.get("classification"),
                "published": meta.get("published"),
                "source": meta.get("source", "arxiv"),
            },
        )
        return int(cur.lastrowid), True


def get_paper(paper_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    return _paper_row_to_dict(row) if row else None


def find_paper_by_arxiv_id(arxiv_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
    return _paper_row_to_dict(row) if row else None


def find_paper_by_github_url(url: str) -> dict | None:
    """按 GitHub 仓库地址反查论文（容忍存储时的尾部差异）。"""
    url = url.strip().rstrip("/")
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM papers WHERE github_url LIKE ?", (f"{url}%",)
        ).fetchone()
    return _paper_row_to_dict(row) if row else None


def list_papers(status: str | None = None) -> list[dict]:
    sql = "SELECT * FROM papers"
    args: tuple = ()
    if status:
        sql += " WHERE status = ?"
        args = (status,)
    sql += " ORDER BY id DESC"
    with get_conn() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [_paper_row_to_dict(r) for r in rows]


def update_paper(paper_id: int, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE papers SET {sets}, updated_at = datetime('now') WHERE id = ?",
            (*fields.values(), paper_id),
        )


def _paper_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("authors", "categories", "keywords"):
        try:
            d[key] = json.loads(d[key] or "[]")
        except (json.JSONDecodeError, TypeError):
            d[key] = []
    return d


# ---------- chunks ----------

@dataclass
class Chunk:
    paper_id: int
    seq: int
    content: str
    heading: str | None = None
    embedding: bytes | None = None
    id: int | None = None


def replace_chunks(paper_id: int, chunks: list[Chunk]) -> None:
    """重建某论文的全部分块（含 embedding）。"""
    with get_conn() as conn:
        conn.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
        conn.executemany(
            "INSERT INTO chunks (paper_id, seq, heading, content, embedding) VALUES (?, ?, ?, ?, ?)",
            [(paper_id, c.seq, c.heading, c.content, c.embedding) for c in chunks],
        )


def get_chunks(paper_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE paper_id = ? ORDER BY seq", (paper_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def all_chunks() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chunks ORDER BY paper_id, seq"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------- innovations ----------

def save_innovations(paper_id: int, items: list[dict]) -> int:
    """覆盖写入某论文的创新点，返回条数。"""
    with get_conn() as conn:
        conn.execute("DELETE FROM innovations WHERE paper_id = ?", (paper_id,))
        conn.executemany(
            """INSERT INTO innovations
               (paper_id, kind, title, description, novelty, source_chunks, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    paper_id,
                    it["kind"],
                    it["title"],
                    it["description"],
                    it.get("novelty"),
                    json.dumps(it.get("source_chunk_ids", []), ensure_ascii=False),
                    json.dumps(it, ensure_ascii=False),
                )
                for it in items
            ],
        )
    return len(items)


def get_innovations(paper_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM innovations WHERE paper_id = ? ORDER BY id", (paper_id,)
        ).fetchall()
    return [_innov_row_to_dict(r) for r in rows]


def _innov_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["source_chunk_ids"] = json.loads(d.pop("source_chunks") or "[]")
    except (json.JSONDecodeError, TypeError):
        d["source_chunk_ids"] = []
    return d


# ---------- ideas ----------

def save_ideas(items: list[dict]) -> list[int]:
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO ideas
               (title, hypothesis, combination, source_innovations, source_papers,
                feasibility, risks, experiments, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    it["title"],
                    it["hypothesis"],
                    it["combination"],
                    json.dumps(it.get("source_innovation_ids", []), ensure_ascii=False),
                    json.dumps(it.get("source_paper_ids", []), ensure_ascii=False),
                    it.get("feasibility"),
                    json.dumps(it.get("risks", []), ensure_ascii=False),
                    json.dumps(it.get("experiments", []), ensure_ascii=False),
                    json.dumps(it, ensure_ascii=False),
                )
                for it in items
            ],
        )
    # executemany 的 lastrowid 不可靠，按插入顺序取最后 N 条 id
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id FROM ideas ORDER BY id DESC LIMIT ?", (len(items),)
        ).fetchall()
    return [r["id"] for r in reversed(rows)]


def list_ideas() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM ideas ORDER BY id DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for key in ("source_innovations", "source_papers", "risks", "experiments"):
            try:
                d[key.replace("source_", "source_")] = json.loads(d[key] or "[]")
            except (json.JSONDecodeError, TypeError):
                pass
        out.append(d)
    return out


# ---------- qa_log ----------

def log_qa(query: str, answer: str, citations: list[dict]) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO qa_log (query, answer, citations) VALUES (?, ?, ?)",
            (query, answer, json.dumps(citations, ensure_ascii=False)),
        )


# ---------- meta ----------

def get_meta(key: str) -> str | None:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(key: str, value: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def get_meta_int(key: str, default: int = 0) -> int:
    try:
        return int(get_meta(key) or default)
    except ValueError:
        return default

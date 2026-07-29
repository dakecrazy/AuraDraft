"""SQLite connection manager with WAL mode and schema initialization.

All schema definitions live here so that every component sees the same
table layout.  Use the *transaction* context manager for atomic writes.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


SCHEMA_SQL = """
-- 文档元数据
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    file_path   TEXT,
    category    TEXT,
    total_tokens INTEGER,
    chunk_count INTEGER,
    tags        TEXT,       -- JSON list
    summary     TEXT,
    created_at  TEXT        -- ISO-8601
);

-- Layer 1: 倒排索引（行级，非 JSON blob）
-- token_id 是规范 tokenizer 的整数 ID（< 2^32，安全）
CREATE TABLE IF NOT EXISTS token_postings (
    token_id    INTEGER,
    doc_id      TEXT,
    chunk_id    INTEGER,
    position    INTEGER,
    frequency   INTEGER DEFAULT 1,
    PRIMARY KEY (token_id, doc_id, chunk_id, position)
);
CREATE INDEX IF NOT EXISTS idx_postings_token
    ON token_postings(token_id);

-- Layer 2: bigram 索引（确定性复合主键，避免 hash() 跨进程不稳定）
CREATE TABLE IF NOT EXISTS bigram_postings (
    t1          INTEGER,
    t2          INTEGER,
    doc_id      TEXT,
    chunk_id    INTEGER,
    position    INTEGER,
    PRIMARY KEY (t1, t2, doc_id, chunk_id, position)
);
CREATE INDEX IF NOT EXISTS idx_bigram_pair
    ON bigram_postings(t1, t2);

-- Chunk token 序列（紧凑二进制存储）
CREATE TABLE IF NOT EXISTS chunk_tokens (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT,
    chunk_idx   INTEGER,
    token_ids   BLOB
);

-- 簇质心（JSON blob — 始终整体读写，行级无意义）
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id   TEXT PRIMARY KEY,
    label        TEXT,
    centroid     TEXT,       -- JSON: {"token_id": weight, ...}
    member_docs  TEXT,       -- JSON list of doc_id strings
    knowledge_card TEXT DEFAULT '',
    doc_count    INTEGER,
    created_at   TEXT,
    last_updated TEXT
);

-- 全局文档频率（用于 TF-IDF 签名）
CREATE TABLE IF NOT EXISTS token_doc_freq (
    token_id   INTEGER PRIMARY KEY,
    doc_count  INTEGER DEFAULT 1
);

-- 文档签名（避免上下文爆炸 — 存 DB，不返回给 Hermes）
CREATE TABLE IF NOT EXISTS doc_signatures (
    doc_id      TEXT PRIMARY KEY,
    signature   TEXT,       -- JSON: {"token_id": weight, ...}
    created_at  TEXT
);

-- 全局统计（替代硬编码 avg_dl）
CREATE TABLE IF NOT EXISTS global_stats (
    key         TEXT PRIMARY KEY,
    value       REAL
);
"""


class Database:
    """WAL-mode SQLite database with schema auto-initialization."""

    def __init__(self, db_path: str | Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # ── connection management ─────────────────────────────────────

    def connect(self) -> sqlite3.Connection:
        """Open (or return) the connection with WAL mode and schema."""
        if self._conn is not None:
            return self._conn
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()
        self._migrate()
        return self._conn

    def _migrate(self) -> None:
        """Run schema migrations based on stored version."""
        current = int(self.get_stat("schema_version", 0.0))
        if current < 1:
            # v1: initial schema
            self.set_stat("schema_version", 1.0)
            current = 1
        if current < 2:
            # v2: add knowledge_card column to clusters table
            try:
                self._conn.execute(
                    "ALTER TABLE clusters ADD COLUMN knowledge_card TEXT DEFAULT ''"
                )
            except sqlite3.OperationalError:
                pass  # column already exists
            self.set_stat("schema_version", 2.0)
            current = 2
        if current < 3:
            # v3: add doc_signatures table (already in SCHEMA_SQL)
            self.set_stat("schema_version", 3.0)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def path(self) -> Path:
        return self._path

    # ── transactions ──────────────────────────────────────────────

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Atomic transaction context manager.

        Rolls back on exception, commits on success.
        """
        conn = self.connect()
        conn.execute("BEGIN")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ── global stats helpers ──────────────────────────────────────

    def get_stat(self, key: str, default: float = 0.0) -> float:
        conn = self.connect()
        row = conn.execute(
            "SELECT value FROM global_stats WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_stat(self, key: str, value: float) -> None:
        conn = self.connect()
        conn.execute(
            "INSERT OR REPLACE INTO global_stats (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()

    def update_avg_dl(self, new_doc_length: int) -> float:
        """Incrementally maintain the average document length.

        Called *after* the document is already committed, so
        ``SELECT COUNT(*)`` already includes the new doc.
        The formula is:
            new_avg = old_avg + (new_len - old_avg) / n
        where n is the total number of documents (including the new one).
        """
        conn = self.connect()
        n = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        old_avg = self.get_stat("avg_dl", 0.0)
        new_avg = old_avg + (new_doc_length - old_avg) / n if n > 0 else float(new_doc_length)
        self.set_stat("avg_dl", new_avg)
        return new_avg

    def total_docs(self) -> int:
        conn = self.connect()
        return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
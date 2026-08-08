#!/usr/bin/env python3
"""One-by-one KB batch ingest with dedup — prevents context/cache explosion.

Mechanical layer only: scan a directory, dedup by MD5, skip already-ingested
paths, then for each unique file run ingest + prefilter. Emits a structured
JSON summary so the agent can make routing decisions in bounded batches
(assign/create per doc) instead of loading everything into one context turn.

Usage:
    bin/kb-python scripts/kb_batch_ingest.py <dir> [--ext pdf,md,txt,docx]

Output (stdout, one JSON object):
    {"files": N, "skipped_dup": N, "skipped_existing": N, "skipped_unsupported": N,
     "ingested": [{"doc_id", "file", "total_tokens", "chunk_count", "candidates": [...]}]}

Runs strictly one-at-a-time (sequential, no parallelism). Strips PYTHONPATH
so tiktoken loads from the kb-agent venv, not Hermes'.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

# --- self-locating: resolve skill dir from this file's location ---
SKILL_DIR = Path(__file__).resolve().parent.parent  # scripts/ -> repo root
sys.path.insert(0, str(SKILL_DIR / "src"))

from kb_agent.tools import init_kb  # noqa: E402
from kb_agent.tools.ops import kb_ingest, kb_prefilter  # noqa: E402

DB = Path(os.environ.get("KB_AGENT_DB", str(Path.home() / ".kb-agent" / "kb_index.db")))
ARCHIVE = Path(os.environ.get("KB_AGENT_ARCHIVE", str(Path.home() / ".kb-agent" / "knowledge_base")))
SUPPORTED = {".pdf", ".md", ".txt", ".docx", ".py"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def existing_paths(db: Path) -> set[str]:
    conn = sqlite3.connect(str(db))
    try:
        return {r[0] for r in conn.execute("SELECT file_path FROM documents")}
    finally:
        conn.close()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: kb_batch_ingest.py <dir> [--ext pdf,md,txt,docx]", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    exts = {("." + e.strip().lstrip(".")).lower() for e in
            (sys.argv[sys.argv.index("--ext") + 1] if "--ext" in sys.argv else "pdf,md,txt,docx,py").split(",")}

    if not root.is_dir():
        print(json.dumps({"error": f"not a directory: {root}"}))
        sys.exit(1)

    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )

    known = existing_paths(DB)
    seen_md5: dict[str, Path] = {}
    stats = {"files": len(files), "skipped_dup": 0, "skipped_existing": 0,
             "skipped_unsupported": 0, "ingested": []}

    session = init_kb(db_path=str(DB), archive_root=str(ARCHIVE))
    session.connect()

    try:
        for f in files:
            # 1. MD5 dedup (handles 3 copies of same file)
            digest = md5(f)
            if digest in seen_md5:
                stats["skipped_dup"] += 1
                print(f"[skip-dup] {f.name} (same as {seen_md5[digest].name})", file=sys.stderr)
                continue
            seen_md5[digest] = f

            # 2. Path dedup against DB
            if str(f) in known:
                stats["skipped_existing"] += 1
                print(f"[skip-existing] {f.name}", file=sys.stderr)
                continue

            # 3. ingest + prefilter (one at a time)
            print(f"[ingest] {f.name} ...", file=sys.stderr)
            try:
                r = kb_ingest(session, str(f))
                doc_id = r["doc_id"]
                cands = kb_prefilter(session, doc_id)
                stats["ingested"].append({
                    "doc_id": doc_id,
                    "file": f.name,
                    "total_tokens": r.get("total_tokens"),
                    "chunk_count": r.get("chunk_count"),
                    "candidates": cands,
                })
                print(f"  -> {doc_id} tok={r.get('total_tokens')} cands={len(cands)}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"[error] {f.name}: {e}", file=sys.stderr)
    finally:
        session.close()

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
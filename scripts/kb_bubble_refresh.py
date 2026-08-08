#!/usr/bin/env python3
"""One-shot KB bubble refresh for Hermes cron.

Runs every cron tick (1 min). Compares the DB mtime against a state file;
regenerates bubble.html + injects meta-refresh ONLY when the DB changed.
Silent (no stdout) when unchanged — so with --no-agent cron mode, an
unchanged tick delivers nothing (no spam).

Because the Hermes cron ticker runs inside the gateway process, this job
starts and stops with Hermes automatically.
"""
import os
import subprocess
import sys
from pathlib import Path

DB = Path.home() / ".kb-agent" / "kb_index.db"
STATE = Path.home() / ".kb-agent" / ".bubble_last_mtime"
VIZ = Path.home() / "kb_agent" / "visualize.py"
VENV_PY = Path.home() / "kb_agent" / ".venv" / "bin" / "python"
REFRESH_INTERVAL = 60  # browser meta-refresh seconds (align with cron 1-min tick)

if not DB.exists():
    sys.exit(0)


def _db_signature(db_path: Path) -> float:
    """Latest mtime across DB + WAL + SHM files.

    SQLite WAL mode writes to the -wal file first; the main .db mtime only
    updates on checkpoint. Checking all three catches real ingest ops that
    never checkpoint.
    """
    candidates = [
        db_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    ]
    return max((f.stat().st_mtime for f in candidates if f.exists()), default=0.0)


sig = _db_signature(DB)

last = None
if STATE.exists():
    try:
        last = float(STATE.read_text().strip())
    except ValueError:
        last = None

# Unchanged → silent exit (no regeneration, no delivery)
if last == sig:
    sys.exit(0)

# DB changed → regenerate (visualize.py injects meta-refresh natively)
# CRITICAL: strip PYTHONPATH so the subprocess loads tiktoken from the
# kb-agent venv, not Hermes' venv (circular-import crash otherwise).
clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
result = subprocess.run(
    [
        str(VENV_PY), str(VIZ), "--mode", "bubble", "--db", str(DB),
        "--refresh-interval", str(REFRESH_INTERVAL),
    ],
    capture_output=True,
    text=True,
    env=clean_env,
)

# Guard: if visualize.py crashed, do NOT advance state — next tick retries.
# Print to stdout so no-agent cron delivers the error (visible in cron history).
if result.returncode != 0:
    print(
        f"⚠ bubble regeneration failed (exit {result.returncode}): "
        f"{result.stderr[-300:]}"
    )
    sys.exit(1)

STATE.write_text(str(sig))
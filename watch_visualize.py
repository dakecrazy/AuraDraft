#!/usr/bin/env python3
"""Watch kb_index.db and regenerate bubble.html on change.

Injects <meta http-equiv="refresh"> into the generated HTML so a browser
tab left open auto-reloads every N seconds — live bubble refresh without
touching visualize.py's fragile nested f-strings.

Usage:
    python watch_visualize.py [--db PATH] [--interval SECONDS] [--mode bubble|cards]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent
VIZ = SKILL_DIR / "visualize.py"
VENV_PY = Path(sys.executable)  # this script runs under the kb-agent venv


def regenerate(db_path: Path, mode: str, interval: int) -> None:
    """Run visualize.py as a subprocess, then inject meta-refresh."""
    subprocess.run(
        [str(VENV_PY), str(VIZ), "--mode", mode, "--db", str(db_path)],
        capture_output=True,
        text=True,
    )
    out = db_path.parent / ("bubble.html" if mode == "bubble" else "visualization.html")
    html = out.read_text("utf-8")
    if 'http-equiv="refresh"' not in html:
        html = html.replace(
            '<meta charset="UTF-8">',
            f'<meta charset="UTF-8">\n<meta http-equiv="refresh" content="{interval}">',
            1,  # only first occurrence
        )
        out.write_text(html, "utf-8")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-refresh kb-agent visualization")
    parser.add_argument("--db", default=str(Path.home() / ".kb-agent" / "kb_index.db"))
    parser.add_argument("--interval", type=int, default=5, help="Poll seconds (default 5)")
    parser.add_argument("--mode", choices=["bubble", "cards"], default="bubble")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    last_sig = _db_signature(db_path)
    regenerate(db_path, args.mode, args.interval)
    print(f"👁  Watching {db_path} every {args.interval}s — Ctrl-C to stop.")

    try:
        while True:
            time.sleep(args.interval)
            sig = _db_signature(db_path)
            if sig != last_sig:
                last_sig = sig
                regenerate(db_path, args.mode, args.interval)
                print(f"  ↻ Regenerated at {time.strftime('%H:%M:%S')}")
    except KeyboardInterrupt:
        print("\n  ⏹ watch stopped.")


if __name__ == "__main__":
    main()
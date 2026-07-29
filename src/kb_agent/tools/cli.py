"""CLI wrapper for the 8 atomic kb-agent tools.

Invoked by Hermes via ``terminal(command="python -m kb_agent.tools.cli ...")``.

Subcommands:
  ingest <file>          — Index a document, return doc_id + signature
  prefilter <doc_id>     — Find Top-K candidate clusters (reads signature from index)
  get-cards <cid> [cid]  — Fetch knowledge_card text for one or more clusters
  assign <doc_id> <cid>  — Assign a document to an existing cluster
  create <label> <doc_id> — Create a new cluster with the document as first member
  update-card <cid>      — Update a cluster's knowledge_card (reads from stdin)
  search <query>         — BM25 search
  archive <file> <label> — Copy file to archive_root/{label}/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src is on the path
_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from kb_agent.tools import (
    init_kb, kb_ingest, kb_prefilter, kb_get_cards,
    kb_assign, kb_create, kb_update_card, kb_search, kb_archive,
)

DB_PATH = "kb_index.db"
ARCHIVE_ROOT = "./knowledge_base"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m kb_agent.tools.cli <subcommand> [args...]", file=sys.stderr)
        print("Subcommands: ingest, prefilter, get-cards, assign, create, update-card, search, archive", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]
    session = init_kb(db_path=DB_PATH, archive_root=ARCHIVE_ROOT)
    session.connect()

    try:
        if cmd == "ingest":
            if len(sys.argv) < 3:
                print("Usage: ingest <file> [doc_id] [category]", file=sys.stderr)
                sys.exit(1)
            file_path = sys.argv[2]
            doc_id = sys.argv[3] if len(sys.argv) > 3 else None
            category = sys.argv[4] if len(sys.argv) > 4 else ""
            result = kb_ingest(session, file_path, doc_id=doc_id, category=category)
            print(json.dumps(result, ensure_ascii=False))

        elif cmd == "prefilter":
            if len(sys.argv) < 3:
                print("Usage: prefilter <doc_id>", file=sys.stderr)
                sys.exit(1)
            doc_id = sys.argv[2]
            result = kb_prefilter(session, doc_id)
            print(json.dumps(result, ensure_ascii=False))

        elif cmd == "get-cards":
            if len(sys.argv) < 3:
                print("Usage: get-cards <cid> [cid ...]", file=sys.stderr)
                sys.exit(1)
            cids = sys.argv[2:]
            result = kb_get_cards(session, cids)
            print(json.dumps(result, ensure_ascii=False))

        elif cmd == "assign":
            if len(sys.argv) < 4:
                print("Usage: assign <doc_id> <cluster_id>", file=sys.stderr)
                print("  card_text (optional) via stdin", file=sys.stderr)
                sys.exit(1)
            doc_id = sys.argv[2]
            cluster_id = sys.argv[3]
            card_text = sys.stdin.read().strip() or None
            result = kb_assign(session, doc_id, cluster_id, card_text=card_text)
            print(json.dumps(result, ensure_ascii=False))

        elif cmd == "create":
            if len(sys.argv) < 4:
                print("Usage: create <label> <doc_id>", file=sys.stderr)
                print("  card_text via stdin", file=sys.stderr)
                sys.exit(1)
            label = sys.argv[2]
            doc_id = sys.argv[3]
            card_text = sys.stdin.read().strip()
            result = kb_create(session, label, card_text, doc_id)
            print(json.dumps(result, ensure_ascii=False))

        elif cmd == "update-card":
            if len(sys.argv) < 3:
                print("Usage: update-card <cluster_id>", file=sys.stderr)
                sys.exit(1)
            cluster_id = sys.argv[2]
            # Read card text from stdin
            card_text = sys.stdin.read().strip()
            if not card_text:
                print(json.dumps({"error": "Card text must be provided via stdin"}))
                sys.exit(1)
            result = kb_update_card(session, cluster_id, card_text)
            print(json.dumps(result, ensure_ascii=False))

        elif cmd == "search":
            if len(sys.argv) < 3:
                print("Usage: search <query> [top_k] [mode]", file=sys.stderr)
                sys.exit(1)
            query = sys.argv[2]
            top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            mode = sys.argv[4] if len(sys.argv) > 4 else "hybrid"
            result = kb_search(session, query, top_k=top_k, mode=mode)
            print(json.dumps(result, ensure_ascii=False))

        elif cmd == "archive":
            if len(sys.argv) < 4:
                print("Usage: archive <file> <label> [doc_id]", file=sys.stderr)
                sys.exit(1)
            file_path = sys.argv[2]
            label = sys.argv[3]
            doc_id = sys.argv[4] if len(sys.argv) > 4 else None
            result = kb_archive(session, file_path, label, doc_id=doc_id)
            print(json.dumps(result, ensure_ascii=False))

        else:
            print(f"Unknown subcommand: {cmd}", file=sys.stderr)
            sys.exit(1)

    finally:
        session.close()


if __name__ == "__main__":
    main()
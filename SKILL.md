---
name: kb-agent
description: "Token-based knowledge base with MoE-inspired routing. 8 atomic CLI tools for document indexing (tiktoken + BM25), statistical clustering (TF-IDF signatures + cosine similarity), knowledge card management, and physical archiving. Zero LLM calls — designed for agent orchestration."
---

# kb-agent

Token-based knowledge base with MoE-inspired routing. Indexes documents using tiktoken's o200k_base tokenizer, builds three-layer token indices (inverted postings + bigram + packed chunks), and clusters them via TF-IDF signature similarity. Exposes 8 atomic CLI tools for agent orchestration — zero LLM calls inside the tools themselves.

## When to use

- Index and search documents by token-level BM25 (not embedding)
- Index PDF documents (pymupdf extraction) alongside text files
- Automatically cluster documents by token-frequency similarity
- Maintain evolving "knowledge cards" per cluster across sessions
- Archive documents into a structured `knowledge_base/{label}/` tree
- Batch ingest 100+ documents with deterministic routing

## Prerequisites

> **`$SKILL_DIR`** throughout this document = the directory containing this `SKILL.md` file. Any agent runtime (OpenClaw, Hermes, standalone) knows where its skills live — substitute that path for `$SKILL_DIR`. No hardcoded paths.

The package is installed into a venv inside the skill directory. **One-time setup** (self-locating, works on any machine / any runtime):

```bash
bash "$SKILL_DIR/setup.sh"
```

This creates `$SKILL_DIR/.venv`, installs the package editable, and prints the exact activation + CLI commands. It respects a `KB_AGENT_PYTHON` env var if you need a specific interpreter. Package installs use a **three-tier fallback**: fast PyPI mirror (Tsinghua/Aliyun) → local Clash/V2Ray proxy (7890/7897/1087/8080) → direct. It also installs **pymupdf** for PDF text extraction (optional — PDFs fall back to PyPDF2 if absent).

After setup, activate the venv:

```bash
source "$SKILL_DIR/.venv/bin/activate"
```

Or call the CLI with the venv Python directly (no activation needed):

```bash
"$SKILL_DIR/.venv/bin/python" -m kb_agent.tools.cli <cmd> <args>
```

## Data paths

- **Database:** defaults to `~/.kb-agent/kb_index.db` — override with the `KB_AGENT_DB` env var (e.g. `KB_AGENT_DB="$SKILL_DIR/kb_index.db"` to keep it beside the skill)
- **Archive:** `./knowledge_base/` (override with `KB_AGENT_ARCHIVE` env var)
- **Token cache:** `.cache/tiktoken/`

> **Note:** the CLI's default DB path is `~/.kb-agent/kb_index.db`, **not** the current working directory. If you want the DB stored alongside the skill, set `KB_AGENT_DB` explicitly. The skill directory's own `kb_index.db` is only used when you point `KB_AGENT_DB` at it.

## 8 Atomic Tools

> **Before running any command below**, set `SKILL_DIR` to this skill's directory (the one containing this `SKILL.md`), e.g.:
> ```bash
> SKILL_DIR=~/.hermes/skills/data-science/kb-agent
> ```
> Substitute the real path for your runtime (OpenClaw, Hermes, standalone). Do **not** copy `$SKILL_DIR` literally — the shell will error with `unbound variable` under `set -u`.

All tools via CLI: `"$SKILL_DIR/bin/kb" <cmd> <args>` — the `bin/kb` wrapper strips any polluted `PYTHONPATH` from the parent environment, so it works on any runtime (Hermes, OpenClaw, standalone) without activation.

### 1. ingest — Index a document

```bash
"$SKILL_DIR/bin/kb" ingest <file> [doc_id] [category]
```

Tokenizes, chunks (256 tokens, 32 overlap), builds inverted + bigram + chunk index. Does NOT classify or assign. Returns `{doc_id, total_tokens, chunk_count, unique_tokens}`.

### 2. prefilter — Statistical pre-screening

```bash
"$SKILL_DIR/bin/kb" prefilter <doc_id>
```

Reads the document's token signature from DB, computes cosine similarity against all cluster centroids, returns Top-K candidates. Returns `[]` if no clusters exist.

```json
[{"cluster_id": "76bf7f85", "label": "深度学习", "similarity": 0.1379, "doc_count": 2}]
```

### 3. get-cards — Read knowledge cards

```bash
"$SKILL_DIR/bin/kb" get-cards <cid> [cid ...]
```

Returns `{cluster_id: card_text}` for the requested clusters.

### 4. assign — Assign document to existing cluster

```bash
cat card.txt | "$SKILL_DIR/bin/kb" assign <doc_id> <cluster_id>
```

Card text from stdin (supports multiline). Updates cluster centroid (running average) and token doc-frequency.

### 5. create — Create new cluster

```bash
cat card.txt | "$SKILL_DIR/bin/kb" create <label> <doc_id>
```

Creates a new cluster with the document as first member. Card text from stdin.

### 6. update-card — Update knowledge card

```bash
echo "new card content" | "$SKILL_DIR/bin/kb" update-card <cluster_id>
```

### 7. search — BM25 retrieval

```bash
"$SKILL_DIR/bin/kb" search <query> [top_k] [mode]
```

Modes: `exact` (inverted index), `phrase` (bigram), `hybrid` (default: 0.6×exact + 0.4×phrase).

### 8. archive — Physical archiving

```bash
"$SKILL_DIR/bin/kb" archive <file> <label> [doc_id]
```

Copies file to `knowledge_base/{label}/`. Handles filename conflicts with doc_id suffix.

## Orchestration workflow

### Standard ingest flow (per document)

```
1. ingest(file)           → doc_id
2. prefilter(doc_id)      → candidates[] or []
3. if candidates:
     get-cards([cids])    → read knowledge cards
     [agent decides: which cluster?]
     assign(doc_id, cid)  ← pipe new card via stdin if updating
   else:
     [agent generates label + card]
     create(label, doc_id) ← pipe card via stdin
4. archive(file, label, doc_id)
```

### Query flow

```
1. search(query)          → ranked results with scores
2. [agent reads chunks via get_chunk_text or opens file directly]
```

### Batch ingest (100+ docs)

For batch, run a Python script in a single process to avoid repeated tiktoken loading (~2s per CLI call). Set `SKILL_DIR` to the skill directory (the one containing this `SKILL.md`), then:

```bash
SKILL_DIR="<path to skill dir>"   # e.g. ~/.hermes/skills/data-science/kb-agent
"$SKILL_DIR/bin/kb-python" - "$SKILL_DIR" <<'PY'
import sys
skill_dir = sys.argv[1]
sys.path.insert(0, skill_dir + "/src")
from kb_agent.tools.session import KnowledgeBaseSession
from kb_agent.tools.ops import kb_ingest, kb_prefilter, kb_assign, kb_create, kb_archive
from pathlib import Path

session = KnowledgeBaseSession("kb_index.db")
session.connect()

for f in Path("./docs/").glob("*.txt"):
    r = kb_ingest(session, str(f))
    doc_id = r["doc_id"]
    candidates = kb_prefilter(session, doc_id)
    if candidates:
        kb_assign(session, doc_id, candidates[0]["cluster_id"])
    else:
        kb_create(session, "新领域", "初始知识档案", doc_id)
    kb_archive(session, str(f), "新领域", doc_id)

session.close()
PY
```

The `"$SKILL_DIR/bin/kb-python" - "$SKILL_DIR"` form passes the skill dir as `sys.argv[1]`, so the script resolves `src/` relative to its own location — no hardcoded paths anywhere. The `kb-python` wrapper strips `PYTHONPATH` pollution, so the batch script's `import tiktoken` loads the venv's own native extension.

### Batch ingest from a directory (one-by-one, cache-safe)

For a whole directory of documents, use the ready-made script — it prevents context/cache explosion by processing strictly one-at-a-time and emitting a compact JSON summary for the agent's routing decisions:

```bash
"$SKILL_DIR/bin/kb-python" "$SKILL_DIR/scripts/kb_batch_ingest.py" <dir> [--ext pdf,md,txt,docx,py]
```

What it does (mechanical layer only — no LLM calls):
- **MD5 dedup** — identical files (e.g. 3 copies of the same PDF) ingest once
- **Path dedup** — skips docs already in the DB (by `file_path`)
- **Sequential ingest + prefilter** — one doc at a time, never loads the whole dir into memory/context
- **Structured JSON output** — `{files, skipped_dup, skipped_existing, ingested: [{doc_id, file, total_tokens, chunk_count, candidates}]}`

The agent then does L1 routing per doc (read candidates + cards → `assign`/`create`), then `archive`. This is the recommended flow for 10+ docs from a folder — it keeps each routing decision in a bounded context turn instead of dumping everything at once.

> **Pitfall:** `kb_ingest` calls `tiktoken.encode` with `disallowed_special=()` (fixed in `canonical.py`), so literal `<|endoftext|>` text in ML/AI papers no longer crashes tokenization.

## Visualization

`visualize.py` generates standalone HTML pages from the knowledge base DB. No server required — any agent or human can open the HTML in a browser.

### Bubble view (default) — interactive D3 force-directed map

```bash
"$SKILL_DIR/bin/kb-python" "$SKILL_DIR/visualize.py" --mode bubble
# → ~/.kb-agent/bubble.html
```

Each cluster is a draggable bubble. Bubble size = doc count. Inside each bubble, top-20 tokens float with gentle animation. Inter-cluster links show cosine similarity. Click a bubble for details, search tokens across all clusters.

### Cards view — static summary

```bash
"$SKILL_DIR/bin/kb-python" "$SKILL_DIR/visualize.py" --mode cards
# → ~/.kb-agent/visualization.html
```

Card-based layout: per-cluster knowledge cards, token signature bars, similarity matrix heatmap, timeline.

### Options

```bash
"$SKILL_DIR/bin/kb-python" "$SKILL_DIR/visualize.py" --db ./custom.db --mode bubble --output ~/viz.html
"$SKILL_DIR/bin/kb-python" "$SKILL_DIR/visualize.py" --mode cards --db ~/.kb-agent/kb_index.db
```

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `~/.kb-agent/kb_index.db` | Database path |
| `--mode` | `bubble` | `bubble` or `cards` |
| `--output` | Auto (alongside DB) | Output HTML path |

### Viewing & live refresh (zero infrastructure)

Generate the bubble with auto-refresh, then open it in the Hermes preview pane:

```bash
"$SKILL_DIR/bin/kb-python" "$SKILL_DIR/visualize.py" --mode bubble --refresh-interval 5
# Then in Hermes: open_preview("file:///Users/<user>/.kb-agent/bubble.html")
```

The `--refresh-interval N` flag injects `<meta http-equiv="refresh" content="N">`.
The Hermes preview pane (or any browser tab) auto-reloads every N seconds.
When you classify new docs, just re-run `visualize.py` — the open pane refreshes
automatically. **No server, no daemon, no port.**

### Auto refresh (optional, paused by default)

For always-on auto-regeneration, the cron module detects DB changes (WAL-aware:
`.db`/`-wal`/`-shm`) and re-runs `visualize.py` automatically:

```bash
hermes cron resume ff4e8bab342d   # the "KB Bubble Refresh" job
```

It regenerates `bubble.html` only on change and injects meta-refresh. Silent when
unchanged → no delivery spam. Guarded so a `visualize.py` crash does not advance
the state file (next tick retries). Uses the `bin/kb-python` wrapper so tiktoken
loads from the kb-agent venv, not Hermes' venv (PYTHONPATH pollution fix).

Works with any agent runtime (OpenClaw, Hermes, standalone Python). Generated HTML is self-contained — only external dependency is the D3.js CDN.

## Key constraints

- **ingest → must assign or create** — otherwise document has index but no cluster, won't appear in search
- **Signature stays in DB** — `kb_prefilter` reads it internally, never returned to caller
- **archive copies, not moves** — source file preserved
- **SQLite WAL mode** — avoid concurrent writes from multiple processes
- **CLI cold start ~2s** — tiktoken BPE loading; use Python script for batch

## Architecture

```
Agent (reasoning layer)
  │
  ├── kb_ingest(file)        → doc_id
  ├── kb_prefilter(doc_id)   → candidate clusters (L0: statistical)
  ├── kb_get_cards(cids)     → knowledge card texts
  │
  ├── [Agent decides: assign or create]
  │
  ├── kb_assign(doc_id, cid) / kb_create(label, card, doc_id)
  ├── kb_update_card(cid, card_text)
  ├── kb_search(query)       → BM25 ranked results
  └── kb_archive(file, label) → archived_path
```

Three-layer index:
- **Layer 1:** Token inverted index (BM25 scoring, K1=1.2, B=0.75)
- **Layer 2:** Bigram phrase index (deterministic composite keys)
- **Layer 3:** Packed chunk token sequences (struct-encoded binary)

MoE routing principle:
- **L0 (free, ms):** Token frequency signature → cosine similarity → Top-K candidates
- **L1 (LLM, tokens):** Agent reads Top-K knowledge cards → semantic judgment
- Scales to 10000+ clusters with bounded compute

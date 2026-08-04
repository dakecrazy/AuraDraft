---
name: kb-agent
description: "Token-based knowledge base with MoE-inspired routing. 8 atomic CLI tools for document indexing (tiktoken + BM25), statistical clustering (TF-IDF signatures + cosine similarity), knowledge card management, and physical archiving. Zero LLM calls — designed for agent orchestration."
---

# kb-agent

Token-based knowledge base with MoE-inspired routing. Indexes documents using tiktoken's o200k_base tokenizer, builds three-layer token indices (inverted postings + bigram + packed chunks), and clusters them via TF-IDF signature similarity. Exposes 8 atomic CLI tools for agent orchestration — zero LLM calls inside the tools themselves.

## When to use

- Index and search documents by token-level BM25 (not embedding)
- Automatically cluster documents by token-frequency similarity
- Maintain evolving "knowledge cards" per cluster across sessions
- Archive documents into a structured `knowledge_base/{label}/` tree
- Batch ingest 100+ documents with deterministic routing

## Prerequisites

The package is installed in a venv at the skill directory. Activate it before use:

```bash
source /home/dakecrazy/.openclaw/workspace-coding/skills/kb-agent/.venv/bin/activate
```

Or call the CLI with the venv Python directly:

```bash
/home/dakecrazy/.openclaw/workspace-coding/skills/kb-agent/.venv/bin/python -m kb_agent.tools.cli <cmd> <args>
```

## Data paths

- **Database:** `kb_index.db` (relative to working directory — set with `KB_AGENT_DB` env var to override)
- **Archive:** `./knowledge_base/` (override with `KB_AGENT_ARCHIVE` env var)
- **Token cache:** `.cache/tiktoken/`

Default DB location is the current working directory. For consistency, set `KB_AGENT_DB` to an absolute path.

## 8 Atomic Tools

All tools via CLI: `python -m kb_agent.tools.cli <cmd> <args>`

### 1. ingest — Index a document

```bash
python -m kb_agent.tools.cli ingest <file> [doc_id] [category]
```

Tokenizes, chunks (256 tokens, 32 overlap), builds inverted + bigram + chunk index. Does NOT classify or assign. Returns `{doc_id, total_tokens, chunk_count, unique_tokens}`.

### 2. prefilter — Statistical pre-screening

```bash
python -m kb_agent.tools.cli prefilter <doc_id>
```

Reads the document's token signature from DB, computes cosine similarity against all cluster centroids, returns Top-K candidates. Returns `[]` if no clusters exist.

```json
[{"cluster_id": "76bf7f85", "label": "深度学习", "similarity": 0.1379, "doc_count": 2}]
```

### 3. get-cards — Read knowledge cards

```bash
python -m kb_agent.tools.cli get-cards <cid> [cid ...]
```

Returns `{cluster_id: card_text}` for the requested clusters.

### 4. assign — Assign document to existing cluster

```bash
cat card.txt | python -m kb_agent.tools.cli assign <doc_id> <cluster_id>
```

Card text from stdin (supports multiline). Updates cluster centroid (running average) and token doc-frequency.

### 5. create — Create new cluster

```bash
cat card.txt | python -m kb_agent.tools.cli create <label> <doc_id>
```

Creates a new cluster with the document as first member. Card text from stdin.

### 6. update-card — Update knowledge card

```bash
echo "new card content" | python -m kb_agent.tools.cli update-card <cluster_id>
```

### 7. search — BM25 retrieval

```bash
python -m kb_agent.tools.cli search <query> [top_k] [mode]
```

Modes: `exact` (inverted index), `phrase` (bigram), `hybrid` (default: 0.6×exact + 0.4×phrase).

### 8. archive — Physical archiving

```bash
python -m kb_agent.tools.cli archive <file> <label> [doc_id]
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

For batch, run a Python script in a single process to avoid repeated tiktoken loading (~2s per CLI call):

```python
import subprocess, sys
script = '''
import sys
sys.path.insert(0, "/home/dakecrazy/.openclaw/workspace-coding/skills/kb-agent/src")
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
'''
subprocess.run(["/home/dakecrazy/.openclaw/workspace-coding/skills/kb-agent/.venv/bin/python", "-c", script])
```

## Visualization

`visualize.py` generates standalone HTML pages from the knowledge base DB. No server required — any agent or human can open the HTML in a browser.

### Bubble view (default) — interactive D3 force-directed map

```bash
python visualize.py --mode bubble
# → ~/.kb-agent/bubble.html
```

Each cluster is a draggable bubble. Bubble size = doc count. Inside each bubble, top-20 tokens float with gentle animation. Inter-cluster links show cosine similarity. Click a bubble for details, search tokens across all clusters.

### Cards view — static summary

```bash
python visualize.py --mode cards
# → ~/.kb-agent/visualization.html
```

Card-based layout: per-cluster knowledge cards, token signature bars, similarity matrix heatmap, timeline.

### Options

```bash
python visualize.py --db ./custom.db --mode bubble --output ~/viz.html
python visualize.py --mode cards --db ~/.kb-agent/kb_index.db
```

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `~/.kb-agent/kb_index.db` | Database path |
| `--mode` | `bubble` | `bubble` or `cards` |
| `--output` | Auto (alongside DB) | Output HTML path |

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

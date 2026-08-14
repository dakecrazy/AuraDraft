#!/usr/bin/env python3
"""AuraDraft visualization — generates standalone HTML pages.

Usage:
    python visualize.py [--db PATH] [--output PATH] [--mode MODE]

Modes:
    cards   — Static card-based layout with token bars, similarity matrix, timeline
    bubble  — Interactive D3 force-directed bubble map with Aura field (default)

Defaults:
    --db      ~/.kb-agent/kb_index.db
    --output  ~/.kb-agent/visualization.html (cards) or ~/.kb-agent/bubble.html (bubble)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure kb_agent is importable
SKILL_DIR = Path(__file__).resolve().parent
_src = SKILL_DIR / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def get_db_path() -> str:
    return os.environ.get("KB_AGENT_DB", str(Path.home() / ".kb-agent" / "kb_index.db"))


# ═══════════════════════════════════════════════════════════
# Data loading (shared)
# ═══════════════════════════════════════════════════════════

def load_clusters(conn):
    """Load all clusters from the DB."""
    rows = conn.execute("SELECT * FROM clusters").fetchall()
    clusters = []
    for row in rows:
        clusters.append({
            "cluster_id": row["cluster_id"],
            "label": row["label"] or "",
            "centroid": {int(k): v for k, v in json.loads(row["centroid"] or "{}").items()},
            "member_doc_ids": json.loads(row["member_docs"] or "[]"),
            "knowledge_card": row["knowledge_card"] or "",
            "doc_count": row["doc_count"] or 0,
            "created_at": row["created_at"] or "",
            "last_updated": row["last_updated"] or "",
        })
    return clusters


def decode_top_tokens(tokenizer, centroid, top_k=20):
    """Decode the top-K weighted tokens from a cluster centroid."""
    sorted_tokens = sorted(centroid.items(), key=lambda x: -x[1])[:top_k]
    result = []
    for tid, weight in sorted_tokens:
        try:
            decoded = tokenizer.decode([int(tid)])
        except Exception:
            decoded = f"<id:{tid}>"
        decoded = decoded.strip().replace("\n", " ").replace("\ufffd", "�")
        if not decoded:
            decoded = f"<id:{tid}>"
        result.append({"token_id": int(tid), "weight": round(weight, 4), "decoded": decoded})
    return result


def compute_cluster_similarity_matrix(clusters):
    """Compute pairwise cosine similarity between cluster centroids."""
    n = len(clusters)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            elif j > i:
                ci = clusters[i].get("centroid", {})
                cj = clusters[j].get("centroid", {})
                common = set(ci.keys()) & set(cj.keys())
                if not common:
                    matrix[i][j] = 0.0
                else:
                    dot = sum(ci[k] * cj[k] for k in common)
                    norm_i = math.sqrt(sum(v ** 2 for v in ci.values()))
                    norm_j = math.sqrt(sum(v ** 2 for v in cj.values()))
                    if norm_i == 0 or norm_j == 0:
                        matrix[i][j] = 0.0
                    else:
                        matrix[i][j] = round(dot / (norm_i * norm_j), 4)
                matrix[j][i] = matrix[i][j]
    return matrix


def get_document_stats(conn):
    """Get per-cluster document statistics."""
    stats = {}
    rows = conn.execute("""
        SELECT d.category, COUNT(*) as doc_count, SUM(d.total_tokens) as total_tokens
        FROM documents d
        GROUP BY d.category
    """).fetchall()
    for row in rows:
        stats[row["category"] or "(uncategorized)"] = {
            "doc_count": row["doc_count"],
            "total_tokens": row["total_tokens"],
        }
    return stats


def get_timeline(conn):
    """Get cluster creation/update timeline."""
    rows = conn.execute("SELECT * FROM clusters ORDER BY created_at").fetchall()
    events = []
    for row in rows:
        events.append({
            "cluster_id": row["cluster_id"],
            "label": row["label"] or "",
            "created_at": row["created_at"] or "",
            "last_updated": row["last_updated"] or "",
            "doc_count": row["doc_count"] or 0,
        })
    return events


def get_overall_stats(conn):
    total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    total_tokens = conn.execute("SELECT COALESCE(SUM(total_tokens), 0) FROM documents").fetchone()[0]
    unique_tokens = conn.execute("SELECT COUNT(DISTINCT token_id) FROM token_postings").fetchone()[0]
    return {
        "total_documents": total_docs,
        "total_clusters": conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0],
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
    }


# ═══════════════════════════════════════════════════════════
# Mode 1: Cards view (original)
# ═══════════════════════════════════════════════════════════

def generate_cards_html(clusters, token_details, sim_matrix, doc_stats, timeline, stats):
    """Generate standalone HTML card-based visualization."""
    cluster_cards = []
    for i, (cluster, tokens) in enumerate(zip(clusters, token_details)):
        cid = cluster.get("cluster_id", "")
        label = cluster.get("label", "")
        doc_count = cluster.get("doc_count", 0)
        card_text = cluster.get("knowledge_card", "(暂无知识档案)")
        created = cluster.get("created_at", "")[:19]
        updated = cluster.get("last_updated", "")[:19]
        centroid_size = len(cluster.get("centroid", {}))

        token_bars = []
        max_weight = max((t["weight"] for t in tokens), default=1)
        for t in tokens:
            bar_width = (t["weight"] / max_weight * 100) if max_weight > 0 else 0
            decoded = t["decoded"].replace("<", "&lt;").replace(">", "&gt;")
            token_bars.append(f"""
                <div class="token-row">
                    <span class="token-text" title="token_id: {t['token_id']}">{decoded}</span>
                    <div class="token-bar-bg">
                        <div class="token-bar-fill" style="width: {bar_width:.1f}%"></div>
                    </div>
                    <span class="token-weight">{t['weight']:.4f}</span>
                </div>""")

        card_escaped = card_text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

        cluster_cards.append(f"""
        <div class="cluster-card" id="cluster-{cid}">
            <div class="cluster-header">
                <h3>{label}</h3>
                <span class="cluster-id">{cid}</span>
            </div>
            <div class="cluster-meta">
                <span>📄 {doc_count} docs</span>
                <span>🧩 {centroid_size} centroid tokens</span>
                <span>📅 {created}</span>
                <span>✏️ {updated}</span>
            </div>
            <div class="cluster-card-text">
                <h4>Knowledge Card</h4>
                <div class="card-content">{card_escaped}</div>
            </div>
            <div class="cluster-tokens">
                <h4>Top-20 Token Signature</h4>
                {''.join(token_bars)}
            </div>
        </div>""")

    sim_html = ""
    if sim_matrix and len(sim_matrix) > 1:
        labels = [c.get("label", c.get("cluster_id", ""))[:12] for c in clusters]
        header = "".join(f"<th>{l}</th>" for l in labels)
        rows_html = ""
        for i, label in enumerate(labels):
            cells = ""
            for j in range(len(labels)):
                val = sim_matrix[i][j]
                color = f"hsl({120 - val * 120}, 70%, {90 - val * 30}%)"
                cells += f'<td style="background: {color}" title="{val}">{val:.2f}</td>'
            rows_html += f"<tr><th>{label}</th>{cells}</tr>"
        sim_html = f"""
        <div class="section">
            <h2>🔗 Cluster Similarity Matrix</h2>
            <table class="sim-table">
                <thead><tr><th></th>{header}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>"""

    timeline_html = ""
    if timeline:
        timeline_items = []
        for ev in timeline:
            label = ev["label"] or ev["cluster_id"]
            time = ev["created_at"][:19] if ev["created_at"] else "?"
            timeline_items.append(f"""
                <div class="timeline-item">
                    <span class="timeline-time">{time}</span>
                    <span class="timeline-label">{label}</span>
                    <span class="timeline-docs">{ev['doc_count']} docs</span>
                </div>""")
        timeline_html = f"""
        <div class="section">
            <h2>📅 Cluster Timeline</h2>
            <div class="timeline">{''.join(timeline_items)}</div>
        </div>"""

    stats_html = f"""
    <div class="stats-bar">
        <div class="stat-item"><span class="stat-num">{stats['total_documents']}</span><span class="stat-label">Documents</span></div>
        <div class="stat-item"><span class="stat-num">{stats['total_clusters']}</span><span class="stat-label">Clusters</span></div>
        <div class="stat-item"><span class="stat-num">{stats['total_tokens']:,}</span><span class="stat-label">Tokens Indexed</span></div>
        <div class="stat-item"><span class="stat-num">{stats['unique_tokens']:,}</span><span class="stat-label">Unique Token Types</span></div>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>kb-agent — Knowledge Base Visualization</title>
    <style>
        :root {{
            --bg: #0f1117; --card-bg: #1a1d27; --card-border: #2a2d3a;
            --text: #e0e0e0; --text-dim: #888; --accent: #4a9eff;
            --accent-dim: #2a6db5; --green: #4ade80; --yellow: #facc15; --red: #f87171;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, 'Segoe UI', 'Noto Sans SC', sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; padding: 20px; }}
        h1 {{ font-size: 1.8rem; margin-bottom: 8px; background: linear-gradient(135deg, var(--accent), var(--green)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .subtitle {{ color: var(--text-dim); margin-bottom: 24px; font-size: 0.9rem; }}
        .stats-bar {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
        .stat-item {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 16px 24px; text-align: center; min-width: 140px; }}
        .stat-num {{ display: block; font-size: 1.8rem; font-weight: 700; color: var(--accent); }}
        .stat-label {{ font-size: 0.8rem; color: var(--text-dim); }}
        .section {{ margin-bottom: 40px; }}
        .section h2 {{ font-size: 1.3rem; margin-bottom: 16px; color: var(--text); }}
        .cluster-card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 16px; padding: 24px; margin-bottom: 24px; }}
        .cluster-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .cluster-header h3 {{ font-size: 1.3rem; color: var(--green); }}
        .cluster-id {{ font-family: monospace; color: var(--text-dim); font-size: 0.85rem; }}
        .cluster-meta {{ display: flex; gap: 16px; font-size: 0.85rem; color: var(--text-dim); margin-bottom: 16px; flex-wrap: wrap; }}
        .cluster-card-text {{ background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; }}
        .cluster-card-text h4 {{ font-size: 0.85rem; color: var(--accent); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .card-content {{ font-size: 0.9rem; color: var(--text); white-space: pre-wrap; }}
        .cluster-tokens h4 {{ font-size: 0.85rem; color: var(--accent); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .token-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }}
        .token-text {{ min-width: 120px; font-family: monospace; font-size: 0.85rem; color: var(--yellow); text-align: right; }}
        .token-bar-bg {{ flex: 1; height: 18px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden; }}
        .token-bar-fill {{ height: 100%; background: linear-gradient(90deg, var(--accent-dim), var(--accent)); border-radius: 4px; transition: width 0.3s; }}
        .token-weight {{ min-width: 60px; font-family: monospace; font-size: 0.8rem; color: var(--text-dim); }}
        .sim-table {{ border-collapse: collapse; font-size: 0.85rem; }}
        .sim-table th, .sim-table td {{ padding: 8px 12px; text-align: center; border: 1px solid var(--card-border); font-family: monospace; }}
        .sim-table th {{ color: var(--text-dim); }}
        .timeline {{ display: flex; flex-direction: column; gap: 8px; }}
        .timeline-item {{ display: flex; gap: 16px; align-items: center; background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 12px 16px; }}
        .timeline-time {{ font-family: monospace; color: var(--accent); font-size: 0.85rem; }}
        .timeline-label {{ font-weight: 600; }}
        .timeline-docs {{ color: var(--text-dim); font-size: 0.85rem; }}
        .empty {{ color: var(--text-dim); font-style: italic; padding: 40px; text-align: center; }}
    </style>
</head>
<body>
    <h1>⚙️ kb-agent Knowledge Base</h1>
    <p class="subtitle">Generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | DB: {get_db_path()}</p>
    {stats_html}
    <div class="section">
        <h2>📚 Clusters</h2>
        {''.join(cluster_cards) if cluster_cards else '<div class="empty">No clusters yet. Ingest documents and create clusters first.</div>'}
    </div>
    {sim_html}
    {timeline_html}
</body>
</html>"""


# ═══════════════════════════════════════════════════════════
# Mode 2: Interactive bubble view (D3 force-directed)
# ═══════════════════════════════════════════════════════════

# Cluster color palette
CLUSTER_COLORS = [
    "#4a9eff", "#50c878", "#ff6b6b", "#ffd93d",
    "#c084fc", "#ff8c42", "#36c5b5", "#f472b6",
    "#60a5fa", "#34d399", "#fb923c", "#a78bfa",
    "#facc15", "#22d3ee", "#fb7185", "#84cc16",
]


def build_bubble_data(clusters, token_details, sim_matrix, stats, tokenizer=None):
    """Build JSON data for the bubble template (planets + Aura layer)."""
    from kb_agent.aura import compute_aura_tokens, aura_summary

    nodes = []
    sim_threshold = 0.05  # only draw links above this similarity

    for i, cluster in enumerate(clusters):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        cid = cluster["cluster_id"]
        label = cluster["label"] or cid
        doc_count = cluster["doc_count"] or 0
        card = cluster["knowledge_card"] or ""

        # Tokens for the bubble interior
        tokens = []
        for t in token_details[i]:
            tokens.append({
                "text": t["decoded"],
                "weight": t["weight"],
            })

        # Associations from similarity matrix
        associations = []
        if sim_matrix:
            for j, other in enumerate(clusters):
                if i == j:
                    continue
                score = sim_matrix[i][j]
                if score >= sim_threshold:
                    associations.append({
                        "clusterId": other["cluster_id"],
                        "label": other["label"] or other["cluster_id"],
                        "score": score,
                    })
            associations.sort(key=lambda x: -x["score"])
            associations = associations[:5]  # top 5

        nodes.append({
            "id": cid,
            "label": label,
            "docCount": doc_count,
            "color": color,
            "card": card,
            "tokens": tokens,
            "associations": associations,
        })

    # ── Aura layer: cross-boundary inspiration tokens ──
    # Use full centroids (not just top-20): cross-boundary signal often
    # lives below the top-K tokens of each cluster.
    centroids = [c.get("centroid", {}) for c in clusters]
    if tokenizer is not None:
        aura_tokens = compute_aura_tokens(
            clusters,
            centroids=centroids,
            decode=lambda tid: tokenizer.decode([int(tid)]),
            top_k=40,
        )
    else:
        aura_tokens = compute_aura_tokens(clusters, token_details, top_k=40)
    summary = aura_summary(aura_tokens, clusters)
    print(f"   Aura: {summary['aura_tokens']} inspiration tokens, "
          f"{len(summary['bridged_pairs'])} bridged cluster pairs")

    return {
        "nodes": nodes,
        "auraTokens": aura_tokens,
        "stats": stats,
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dbPath": get_db_path(),
    }


def generate_bubble_html(data_json):
    """Generate the interactive D3 bubble visualization HTML."""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>KB Cluster Bubble Map</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  background: #0a0e1a;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  overflow: hidden;
  color: #e0e6f0;
  width: 100vw; height: 100vh;
}}
#canvas-container {{ width: 100%; height: 100%; position: relative; }}
svg {{ width: 100%; height: 100%; display: block; }}

#controls {{
  position: absolute; top: 12px; left: 12px; z-index: 10;
  background: rgba(16,22,40,0.92); backdrop-filter: blur(12px);
  border: 1px solid rgba(80,120,200,0.2); border-radius: 12px;
  padding: 12px 16px; display: flex; gap: 10px; align-items: center;
  flex-wrap: wrap; max-width: 420px;
}}
#search-input {{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(100,140,220,0.3); border-radius: 8px;
  padding: 8px 12px; color: #e0e6f0; font-size: 13px; width: 200px;
  outline: none; transition: border-color 0.2s;
}}
#search-input:focus {{ border-color: rgba(100,180,255,0.6); box-shadow: 0 0 0 2px rgba(100,180,255,0.15); }}
#search-input::placeholder {{ color: rgba(160,180,210,0.4); }}
.btn {{
  background: rgba(60,100,180,0.3); border: 1px solid rgba(100,140,220,0.3);
  border-radius: 8px; padding: 6px 14px; color: #a0c4ff; font-size: 12px;
  cursor: pointer; transition: all 0.2s;
}}
.btn:hover {{ background: rgba(60,100,180,0.5); border-color: rgba(100,180,255,0.5); }}
.btn.active {{ background: rgba(80,120,220,0.6); color: #fff; }}

#stats-overlay {{
  position: absolute; top: 12px; right: 12px; z-index: 5;
  background: rgba(16,22,40,0.85); backdrop-filter: blur(8px);
  border: 1px solid rgba(80,120,200,0.15); border-radius: 10px;
  padding: 10px 16px; font-size: 12px; color: rgba(160,180,210,0.7);
  display: flex; gap: 16px; flex-wrap: wrap;
}}
#stats-overlay .stat {{ display: flex; gap: 4px; align-items: center; }}
#stats-overlay .stat-val {{ color: #70b4ff; font-weight: 600; }}

#info-panel {{
  position: absolute; top: 12px; right: 12px; z-index: 10;
  background: rgba(16,22,40,0.94); backdrop-filter: blur(12px);
  border: 1px solid rgba(80,120,200,0.2); border-radius: 12px;
  padding: 18px 22px; width: 320px; max-height: 80vh; overflow-y: auto;
  opacity: 0; transform: translateX(340px); transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}}
#info-panel.visible {{ opacity: 1; transform: translateX(0); }}
#info-panel h3 {{ font-size: 15px; color: #70b4ff; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }}
#info-panel .close-btn {{ cursor: pointer; color: rgba(160,180,210,0.5); font-size: 18px; }}
#info-panel .close-btn:hover {{ color: #fff; }}
#info-panel .meta {{ font-size: 11px; color: rgba(160,180,210,0.6); margin-bottom: 12px; }}
#info-panel .card-text {{ font-size: 13px; line-height: 1.6; color: #c0cad8; white-space: pre-wrap; }}
#info-panel .token-cloud {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(80,120,200,0.15); }}
#info-panel .token-tag {{
  background: rgba(60,100,180,0.2); border: 1px solid rgba(100,140,220,0.2);
  border-radius: 6px; padding: 3px 8px; font-size: 11px; color: #80a8d8; cursor: pointer;
}}
#info-panel .assoc-section {{ margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(80,120,200,0.15); }}
#info-panel .assoc-title {{ font-size: 12px; color: rgba(160,180,210,0.6); margin-bottom: 8px; }}
#info-panel .assoc-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; cursor: pointer; }}
#info-panel .assoc-bar {{ height: 4px; border-radius: 2px; background: linear-gradient(90deg, rgba(100,160,255,0.6), rgba(100,200,200,0.4)); }}

#legend {{
  position: absolute; bottom: 12px; left: 12px; z-index: 10;
  background: rgba(16,22,40,0.85); backdrop-filter: blur(8px);
  border: 1px solid rgba(80,120,200,0.15); border-radius: 10px;
  padding: 10px 14px; font-size: 11px; color: rgba(160,180,210,0.7);
  display: flex; gap: 16px;
}}
#legend .item {{ display: flex; align-items: center; gap: 6px; }}
#legend .dot {{ width: 10px; height: 10px; border-radius: 50%; }}

#toast {{
  position: absolute; bottom: 60px; left: 50%; transform: translateX(-50%);
  background: rgba(16,22,40,0.95); border: 1px solid rgba(100,180,255,0.3);
  border-radius: 10px; padding: 10px 20px; font-size: 13px; color: #a0c4ff;
  opacity: 0; transition: opacity 0.3s; z-index: 20; pointer-events: none;
}}
#toast.visible {{ opacity: 1; }}

#loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); z-index: 5; text-align: center; }}
#loading .spinner {{
  width: 40px; height: 40px; border: 3px solid rgba(100,180,255,0.15);
  border-top-color: rgba(100,180,255,0.7); border-radius: 50%;
  animation: spin 1s linear infinite; margin: 0 auto 12px;
}}
#loading p {{ font-size: 13px; color: rgba(160,180,210,0.5); }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style>
</head>
<body>
<div id="canvas-container">
  <div id="controls">
    <input id="search-input" type="text" placeholder="搜索 token 或概念..." />
    <button class="btn" id="btn-aura" onclick="toggleAura()">Aura 场</button>
    <button class="btn" id="btn-reset" onclick="resetView()">重置视图</button>
  </div>
  <div id="stats-overlay"></div>
  <div id="info-panel">
    <h3 id="panel-title">— <span class="close-btn" onclick="closePanel()">✕</span></h3>
    <div class="meta" id="panel-meta"></div>
    <div class="card-text" id="panel-card"></div>
    <div class="token-cloud" id="panel-tokens"></div>
    <div class="assoc-section" id="panel-assoc">
      <div class="assoc-title">📊 关联簇</div>
      <div id="panel-assoc-list"></div>
    </div>
  </div>
  <div id="legend">
    <div class="item"><div class="dot" style="background:rgba(100,180,255,0.5)"></div>Cluster</div>
    <div class="item"><div class="dot" style="background:rgba(100,255,180,0.5)"></div>Token</div>
    <div class="item"><div class="dot" style="background:rgba(255,180,100,0.5)"></div>关联线</div>
    <div class="item"><div class="dot" style="background:rgba(120,220,200,0.6)"></div>Aura 启发场</div>
    <div class="item">点击气泡查看详情 · 拖拽移动 · 滚轮缩放</div>
  </div>
  <div id="toast"></div>
  <div id="loading"><div class="spinner"></div><p>构建知识图谱中...</p></div>
  <svg id="bubble-svg"></svg>
</div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
// ═══════════════════════════════════════════════════════════
// Injected data from visualize.py
// ═══════════════════════════════════════════════════════════
const KB_DATA = {data_json};

// Stats overlay
(function() {{
  const s = KB_DATA.stats;
  const overlay = document.getElementById('stats-overlay');
  overlay.innerHTML = `
    <div class="stat"><span>📄</span><span class="stat-val">${{s.total_documents}}</span><span>docs</span></div>
    <div class="stat"><span>🧩</span><span class="stat-val">${{s.total_clusters}}</span><span>clusters</span></div>
    <div class="stat"><span>🔢</span><span class="stat-val">${{s.total_tokens.toLocaleString()}}</span><span>tokens</span></div>
  `;
}})();

// ═══════════════════════════════════════════════════════════
// Setup
// ═══════════════════════════════════════════════════════════
const svg = d3.select("#bubble-svg");
const width = window.innerWidth;
const height = window.innerHeight;
svg.attr("viewBox", [0, 0, width, height]);

const defs = svg.append("defs");
defs.append("filter").attr("id","glow")
  .attr("x","-50%").attr("y","-50%").attr("width","200%").attr("height","200%")
  .html(`<feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>`);

KB_DATA.nodes.forEach(c => {{
  const grad = defs.append("radialGradient").attr("id",`grad-${{c.id}}`)
    .attr("cx","35%").attr("cy","35%");
  grad.append("stop").attr("offset","0%").attr("stop-color",c.color).attr("stop-opacity",0.5);
  grad.append("stop").attr("offset","70%").attr("stop-color",c.color).attr("stop-opacity",0.15);
  grad.append("stop").attr("offset","100%").attr("stop-color",c.color).attr("stop-opacity",0.05);
}});

const g = svg.append("g").attr("class","everything");
const zoom = d3.zoom().scaleExtent([0.3,3]).on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

// Build links from associations
const links = [];
KB_DATA.nodes.forEach(c => {{
  c.associations.forEach(a => {{
    const t = KB_DATA.nodes.find(x => x.id === a.clusterId);
    if (t) links.push({{source:c.id, target:a.clusterId, score:a.score}});
  }});
}});

// Radius scale
const maxDocs = d3.max(KB_DATA.nodes, d => d.docCount) || 1;
const radiusScale = d3.scaleSqrt().domain([1,Math.max(maxDocs,2)]).range([55,130]);

// ── Pre-compute token positions (collision-free) ──
function layoutTokens(cluster) {{
  const r = cluster.radius;
  const innerR = r - 20;
  const tokens = cluster.tokens;

  tokens.forEach(t => {{
    t.clusterId = cluster.id;
    const cjk = t.text.match(/[\u4e00-\u9fff]/g);
    const cjkCount = cjk ? cjk.length : 0;
    const asciiCount = t.text.length - cjkCount;
    t.fontSize = 11 + t.weight * 7;
    t.tw = cjkCount * t.fontSize + asciiCount * t.fontSize * 0.6 + 6;
    t.th = t.fontSize + 4;
  }});

  const sorted = [...tokens].sort((a,b) => b.weight - a.weight);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  sorted.forEach((t, i) => {{
    if (i === 0) {{ t.localX = 0; t.localY = 0; return; }}
    const angle = i * goldenAngle;
    const radiusFactor = Math.sqrt(i / sorted.length);
    const rr = innerR * 0.9 * radiusFactor;
    t.localX = rr * Math.cos(angle);
    t.localY = rr * Math.sin(angle);
  }});

  // Collision resolution
  const padding = 4;
  for (let iter = 0; iter < 50; iter++) {{
    for (let i = 0; i < sorted.length; i++) {{
      for (let j = i + 1; j < sorted.length; j++) {{
        const a = sorted[i], b = sorted[j];
        const dx = b.localX - a.localX;
        const dy = b.localY - a.localY;
        const dist = Math.sqrt(dx*dx + dy*dy);
        const minDist = (a.tw + b.tw) / 2 * 0.95 + padding;
        if (dist < minDist && dist > 0.001) {{
          const push = (minDist - dist) / 2;
          const nx = dx / dist;
          const ny = dy / dist;
          a.localX -= nx * push; a.localY -= ny * push;
          b.localX += nx * push; b.localY += ny * push;
        }}
      }}
    }}
    sorted.forEach(t => {{
      const dd = Math.sqrt(t.localX**2 + t.localY**2);
      const maxR = innerR - Math.max(t.tw, t.th) / 2;
      if (dd > maxR && dd > 0.001) {{
        t.localX = (t.localX / dd) * maxR;
        t.localY = (t.localY / dd) * maxR;
      }}
    }});
  }}
  cluster.tokenNodes = sorted;
}}

const nodes = KB_DATA.nodes.map(c => {{
  const r = radiusScale(c.docCount || 1);
  const node = {{...c, radius: r}};
  layoutTokens(node);
  return node;
}});
const nodeMap = new Map(nodes.map(n => [n.id, n]));

// ═══════════════════════════════════════════════════════════
// Draw
// ═══════════════════════════════════════════════════════════
const auraLayer = g.append("g").attr("class","aura-field");          // heatmap (bottom)
const linkGroup = g.append("g").attr("class","links");
const nodeGroup = g.append("g").attr("class","nodes");
const auraTokensLayer = g.append("g").attr("class","aura-tokens");   // drifting tokens (top)

const linkElements = linkGroup.selectAll("line").data(links).enter().append("line")
  .attr("stroke","rgba(255,180,100,0.15)")
  .attr("stroke-width", d => d.score * 3)
  .attr("stroke-dasharray","4 4");

const nodeElements = nodeGroup.selectAll("g.cluster-node")
  .data(nodes, d => d.id).enter().append("g")
  .attr("class","cluster-node")
  .style("cursor","pointer")
  .call(d3.drag().on("start",dragStarted).on("drag",dragged).on("end",dragEnded));

nodeElements.append("circle")
  .attr("class","bubble-bg")
  .attr("r", d => d.radius)
  .attr("fill", d => `url(#grad-${{d.id}})`)
  .attr("stroke", d => d.color)
  .attr("stroke-width", 1.5)
  .attr("stroke-opacity", 0.4)
  .attr("filter","url(#glow)");

nodeElements.append("text")
  .attr("class","cluster-label")
  .attr("text-anchor","middle")
  .attr("dy", d => -d.radius + 20)
  .attr("fill", d => d.color)
  .attr("font-size","14px").attr("font-weight","600")
  .attr("opacity",0.9).text(d => d.label);

nodeElements.append("text")
  .attr("class","doc-count")
  .attr("text-anchor","middle")
  .attr("dy", d => -d.radius + 34)
  .attr("fill","rgba(160,180,210,0.4)")
  .attr("font-size","10px")
  .text(d => `${{d.docCount}} docs`);

// Token text inside bubbles
const tokenGroups = nodeElements.append("g").attr("class","token-group");
tokenGroups.each(function(d) {{
  const tg = d3.select(this);
  d.tokenNodes.forEach((tn) => {{
    tg.append("text")
      .attr("class","token-text")
      .attr("data-token", tn.text)
      .attr("text-anchor","middle")
      .attr("fill", d.color)
      .attr("font-size", tn.fontSize)
      .attr("font-weight", tn.weight > 0.7 ? "600" : "400")
      .attr("opacity", 0.35 + tn.weight * 0.5)
      .attr("data-ox", tn.localX)
      .attr("data-oy", tn.localY)
      .text(tn.text);
  }});
}});

// Click handler
nodeElements.on("click", (event, d) => {{
  event.stopPropagation();
  showPanel(d);
}});
svg.on("click", () => closePanel());

// ═══════════════════════════════════════════════════════════
// Gentle floating animation
// ═══════════════════════════════════════════════════════════
let t0 = 0;
function animateTokens() {{
  t0 += 0.008;
  d3.selectAll(".token-text").each(function() {{
    const el = d3.select(this);
    const ox = parseFloat(el.attr("data-ox"));
    const oy = parseFloat(el.attr("data-oy"));
    const wx = Math.sin(t0 * 1.5 + ox * 0.05) * 2;
    const wy = Math.cos(t0 * 1.2 + oy * 0.05) * 1.5;
    el.attr("x", ox + wx).attr("y", oy + wy);
  }});
  requestAnimationFrame(animateTokens);
}}
animateTokens();

// ═══════════════════════════════════════════════════════════
// Aura field — shared inspiration layer
// H(p) = concentration(p) × entropy(p)
// ═══════════════════════════════════════════════════════════
let auraVisible = true;

function auraIntensity(px, py, nodes) {{
  // concentration: total affinity from all clusters (Gaussian kernels)
  let sumW = 0;
  const ws = [];
  for (const n of nodes) {{
    const dx = px - n.x, dy = py - n.y;
    const sigma = n.radius * 1.6;  // atmosphere thickness scales with planet size
    const w = Math.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma));
    ws.push(w); sumW += w;
  }}
  if (sumW <= 1e-9) return 0;
  // entropy of belonging distribution q_i = w_i / Σw
  let H = 0;
  for (const w of ws) {{
    const q = w / sumW;
    if (q > 1e-9) H -= q * Math.log(q);
  }}
  return sumW * H;
}}

function renderAura() {{
  if (!auraLayer) return;

  // sample the field on a coarse grid
  const step = 24;
  const pts = [];
  const xs = d3.range(0, width + step, step);
  const ys = d3.range(0, height + step, step);
  const vals = [];
  for (const x of xs) for (const y of ys) {{
    const v = auraIntensity(x, y, nodes);
    vals.push(v);
    pts.push({{x, y, v}});
  }}
  const vmax = d3.max(vals) || 1;

  const cell = 26;
  const opacityScale = d3.scaleSqrt().domain([0, vmax]).range([0, 0.32]);
  const auraColor = v => d3.interpolateRgbBasis(["#1a2f5e", "#2b5aa0", "#3fa0d8", "#7fe3d4", "#c8f7e9"])(v / vmax);

  auraLayer.selectAll("rect").remove();
  auraLayer.selectAll("rect").data(pts).enter().append("rect")
    .attr("x", d => d.x - cell/2).attr("y", d => d.y - cell/2)
    .attr("width", cell + 1).attr("height", cell + 1)
    .attr("fill", d => auraColor(d.v))
    .attr("opacity", d => opacityScale(d.v))
    .attr("pointer-events", "none");
}}

function renderAuraTokens() {{
  if (!auraTokensLayer || !KB_DATA.auraTokens || KB_DATA.auraTokens.length === 0) return;
  const auraList = KB_DATA.auraTokens;

  // Position each cross-boundary token at the weighted midpoint of its clusters,
  // offset outward so it sits in the boundary zone rather than planet centers.
  const items = auraList.map((at, idx) => {{
    let sx = 0, sy = 0, sw = 0;
    at.clusterIds.forEach(cid => {{
      const n = nodeMap.get(cid);
      const w = (at.strengths && at.strengths[cid]) || 1;
      if (n) {{ sx += n.x * w; sy += n.y * w; sw += w; }}
    }});
    if (sw <= 0) return null;
    let x = sx / sw, y = sy / sw;
    // push toward cluster-pair boundary: offset away from the strongest cluster
    const strongest = at.clusterIds.reduce((a, b) =>
      ((at.strengths[b] || 0) > (at.strengths[a] || 0) ? b : a));
    const sn = nodeMap.get(strongest);
    if (sn) {{
      let dx = x - sn.x, dy = y - sn.y;
      const d = Math.sqrt(dx*dx + dy*dy) || 1;
      const r = Math.max(28, sn.radius * 0.55);
      x = sn.x + dx / d * (sn.radius + r * 0.35);
      y = sn.y + dy / d * (sn.radius + r * 0.35);
    }}
    const maxW = auraList[0].weight || 1;
    return {{
      text: at.text,
      x, y,
      ox: x, oy: y,
      size: 11 + (at.weight / maxW) * 10,
      opacity: 0.35 + (at.weight / maxW) * 0.5,
      clusters: at.clusterIds,
      weight: at.weight,
      idx,
    }};
  }}).filter(Boolean);

  // Simple collision: spread overlapping aura tokens vertically
  items.sort((a,b) => b.weight - a.weight);
  for (let i = 0; i < items.length; i++) {{
    for (let j = i + 1; j < items.length; j++) {{
      const a = items[i], b = items[j];
      if (Math.abs(a.ox - b.ox) < 60 && Math.abs(a.oy - b.oy) < 16) {{
        b.oy += (b.oy >= a.oy ? 1 : -1) * (16 + Math.random() * 8);
        b.ox += (Math.random() - 0.5) * 30;
      }}
    }}
  }}

  auraTokensLayer.selectAll("text").remove();
  auraTokensLayer.selectAll("text").data(items).enter().append("text")
    .attr("class", "aura-token")
    .attr("text-anchor", "middle")
    .attr("fill", "#a8f0e0")
    .attr("font-size", d => d.size)
    .attr("font-weight", d => d.size > 17 ? "600" : "400")
    .attr("opacity", d => d.opacity)
    .attr("pointer-events", "none")
    .attr("data-ax", d => d.ox)
    .attr("data-ay", d => d.oy)
    .attr("data-phase", d => d.idx * 1.7)
    .text(d => d.text)
    .style("filter", "drop-shadow(0 0 6px rgba(120,220,200,0.5))");
}}

function toggleAura() {{
  auraVisible = !auraVisible;
  const op = auraVisible ? 1 : 0;
  if (auraLayer) auraLayer.style("opacity", op).style("display", auraVisible ? "" : "none");
  if (auraTokensLayer) auraTokensLayer.style("opacity", op).style("display", auraVisible ? "" : "none");
  document.getElementById("btn-aura").classList.toggle("active", auraVisible);
}}

// Gentle drift animation for aura tokens (separate rhythm from planet tokens)
let auraT = 0;
function animateAuraTokens() {{
  auraT += 0.012;
  d3.selectAll(".aura-token").each(function() {{
    const el = d3.select(this);
    const ax = parseFloat(el.attr("data-ax"));
    const ay = parseFloat(el.attr("data-ay"));
    const ph = parseFloat(el.attr("data-phase"));
    const dx = Math.sin(auraT * 0.7 + ph) * 7;
    const dy = Math.cos(auraT * 0.5 + ph * 1.3) * 5;
    el.attr("x", ax + dx).attr("y", ay + dy);
  }});
  requestAnimationFrame(animateAuraTokens);
}}


const simulation = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id)
    .distance(d => 200 + (1-d.score)*100).strength(d => d.score * 0.3))
  .force("charge", d3.forceManyBody().strength(-450))
  .force("center", d3.forceCenter(width/2, height/2))
  .force("collision", d3.forceCollide().radius(d => d.radius + 10))
  .on("tick", ticked);

function ticked() {{
  nodes.forEach(n => {{
    n.x = Math.max(n.radius + 10, Math.min(width - n.radius - 10, n.x));
    n.y = Math.max(n.radius + 10, Math.min(height - n.radius - 10, n.y));
  }});
  linkElements
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  nodeElements.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}}

// Render the Aura field once the layout has settled
simulation.on("end", () => {{ renderAura(); renderAuraTokens(); animateAuraTokens(); }});

function dragStarted(event,d) {{ if(!event.active) simulation.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }}
function dragged(event,d) {{ d.fx=event.x; d.fy=event.y; }}
function dragEnded(event,d) {{ if(!event.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }}

// ═══════════════════════════════════════════════════════════
// Info Panel
// ═══════════════════════════════════════════════════════════
function showPanel(d) {{
  const p = document.getElementById("info-panel");
  document.getElementById("panel-title").innerHTML = `${{d.label}} <span class="close-btn" onclick="closePanel()">✕</span>`;
  document.getElementById("panel-meta").textContent = `${{d.docCount}} docs · cluster_id: ${{d.id}}`;
  document.getElementById("panel-card").textContent = d.card || '(暂无知识档案)';
  const tc = document.getElementById("panel-tokens");
  tc.innerHTML = "";
  d.tokens.forEach(t => {{
    const tag = document.createElement("span");
    tag.className = "token-tag";
    tag.textContent = t.text;
    tag.style.fontSize = `${{10 + t.weight * 4}}px`;
    tag.style.opacity = 0.5 + t.weight * 0.5;
    tag.onclick = () => searchToken(t.text);
    tc.appendChild(tag);
  }});
  const al = document.getElementById("panel-assoc-list");
  al.innerHTML = "";
  if (d.associations && d.associations.length > 0) {{
    d.associations.forEach(a => {{
      const item = document.createElement("div");
      item.className = "assoc-item";
      item.innerHTML = `<span class="assoc-label">${{a.label}}</span><div class="assoc-bar" style="width:${{a.score*100}}px"></div><span style="font-size:10px;color:rgba(160,180,210,0.4);margin-left:auto;">${{(a.score*100).toFixed(0)}}%</span>`;
      item.onclick = () => {{
        const tgt = nodes.find(n => n.id === a.clusterId);
        if (tgt) {{ svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity.translate(width/2-tgt.x, height/2-tgt.y)); setTimeout(()=>showPanel(tgt),400); }}
      }};
      al.appendChild(item);
    }});
  }} else {{ al.innerHTML = "<span style='font-size:12px;color:rgba(160,180,210,0.3)'>无关联簇</span>"; }}
  p.classList.add("visible");
}}
function closePanel() {{ document.getElementById("info-panel").classList.remove("visible"); }}

// ═══════════════════════════════════════════════════════════
// Search
// ═══════════════════════════════════════════════════════════
function searchToken(query) {{
  if (!query) return;
  const q = query.toLowerCase().trim();
  let found = false;
  nodeElements.each(function(d) {{
    const node = d3.select(this);
    const match = d.tokens.some(t => t.text.toLowerCase().includes(q)) || d.label.toLowerCase().includes(q);
    if (match) {{
      node.select(".bubble-bg").transition().duration(300).attr("stroke-opacity",0.9).attr("stroke-width",3);
      node.selectAll(".token-text").each(function(tn) {{
        if (tn.text.toLowerCase().includes(q)) d3.select(this).transition().duration(300).attr("fill","#fff").attr("font-weight","700");
      }});
      found = true;
    }} else {{
      node.select(".bubble-bg").transition().duration(300).attr("stroke-opacity",0.15).attr("stroke-width",1);
    }}
  }});
  if (!found) showToast(`未找到包含「${{query}}」的簇`);
}}
document.getElementById("search-input").addEventListener("input", e => {{ if(e.target.value) searchToken(e.target.value); else resetView(); }});

// ═══════════════════════════════════════════════════════════
// Utils
// ═══════════════════════════════════════════════════════════
function resetView() {{
  nodeElements.each(function(d) {{
    d3.select(this).select(".bubble-bg").transition().duration(300)
      .attr("stroke",d.color).attr("stroke-width",1.5).attr("stroke-opacity",0.4);
    d3.select(this).selectAll(".token-text").each(function(tn) {{
      d3.select(this).transition().duration(300)
        .attr("fill",d.color).attr("font-weight",tn.weight>0.7?"600":"400")
        .attr("opacity",0.35+tn.weight*0.5);
    }});
  }});
  document.getElementById("search-input").value = "";
  svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
}}
function showToast(msg) {{
  const t = document.getElementById("toast");
  t.textContent = msg; t.classList.add("visible");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("visible"), 2500);
}}

document.getElementById("info-panel").addEventListener("click", e => e.stopPropagation());
document.getElementById("controls").addEventListener("click", e => e.stopPropagation());

setTimeout(() => {{ document.getElementById("loading").style.display = "none"; }}, 600);
console.log("KB Bubble Map loaded ✓", KB_DATA.stats);
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="kb-agent cluster visualization — generates standalone HTML pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python visualize.py                          # bubble view (default)
  python visualize.py --mode cards             # card-based static view
  python visualize.py --mode bubble --output ~/my-viz.html
  python visualize.py --db ./custom.db --mode bubble
""",
    )
    parser.add_argument("--db", default=get_db_path(), help="Path to kb_index.db")
    parser.add_argument("--output", default=None, help="Output HTML path (default: auto)")
    parser.add_argument(
        "--mode",
        choices=["bubble", "cards"],
        default="bubble",
        help="Visualization mode (default: bubble)",
    )
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=0,
        help="Deprecated no-op. Electron auto-reloads file:// on change, so no "
        "client-side refresh is needed. Kept for CLI backward compatibility.",
    )
    args = parser.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    default_output = (
        str(Path(db_path).parent / "bubble.html")
        if args.mode == "bubble"
        else str(Path(db_path).parent / "visualization.html")
    )
    output_path = args.output or default_output

    # Import kb_agent components
    from kb_agent.storage.db import Database
    from kb_agent.tokenizer.canonical import CanonicalTokenizer

    db = Database(db_path)
    tokenizer = CanonicalTokenizer()
    conn = db.connect()

    # Load data
    clusters = load_clusters(conn)
    token_details = [
        decode_top_tokens(tokenizer, c.get("centroid", {}), top_k=20)
        for c in clusters
    ]
    sim_matrix = compute_cluster_similarity_matrix(clusters) if len(clusters) > 1 else []
    doc_stats = get_document_stats(conn)
    timeline = get_timeline(conn)
    stats = get_overall_stats(conn)

    # Generate HTML
    if args.mode == "bubble":
        data = build_bubble_data(clusters, token_details, sim_matrix, stats, tokenizer=tokenizer)
        html = generate_bubble_html(json.dumps(data, ensure_ascii=False))
    else:
        html = generate_cards_html(
            clusters, token_details, sim_matrix, doc_stats, timeline, stats
        )

    # NOTE: live auto-reload is NOT injected here. Under file:// the browser
    # blocks fetch() (CORS) and document.lastModified is cached at page-load
    # (does not update dynamically), so no client-side JS can detect file
    # changes. Live refresh is handled externally: the Electron webview
    # auto-reloads a file:// page when the file changes on disk, so re-running
    # this script is enough. --refresh-interval is accepted for backward
    # compatibility but is a no-op.

    # Replace CDN D3 with a local relative path. Under file:// the webview
    # CSP blocks external scripts, so the SVG never renders (blank/error).
    # d3.v7.min.js must sit next to the output HTML.
    html = html.replace(
        'src="https://d3js.org/d3.v7.min.js"',
        'src="d3.v7.min.js"',
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")

    print(f"✅ {args.mode} visualization saved to: {output_path}")
    print(f"   Clusters: {len(clusters)} | Documents: {stats['total_documents']} | Tokens: {stats['total_tokens']:,}")

    db.close()


if __name__ == "__main__":
    main()

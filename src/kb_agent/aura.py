"""Aura field — the shared inspiration layer of AuraDraft.

Concept
-------
Clusters are "planets" (stable TF-IDF centroids). The Aura is a continuous
inspiration field that permeates the space *between* clusters and is densest
at cluster boundaries — the places where cross-domain inspiration happens.

Two computations live here:

1. Token aura weight (server-side, statistical):
       token_aura_weight(t) = total_strength(t) × cross_cluster_entropy(t)
   A token appearing in only one cluster has zero cross-entropy → it sinks
   into its planet. A token spanning N clusters has high cross-entropy →
   it drifts into the boundary zone.

2. Field intensity (client-side, from rendered positions):
       H(p) = concentration(p) × entropy(p)
       concentration(p) = Σ_i exp(-||p - c_i||² / 2σ_i²)
       entropy(p) = Shannon entropy of the belonging distribution q_i(p)
   Maximum at boundaries (high concentration AND high belonging entropy).

Zero LLM calls — pure statistics over existing cluster centroids.
"""

from __future__ import annotations

import math


def _is_meaningful_text(text: str) -> bool:
    """Reject empty / single-char / replacement-char garbage tokens."""
    stripped = text.strip()
    if not stripped:
        return False
    if "\ufffd" in text:
        return False
    if len(stripped) < 2:
        return False
    return True


# Compact English stopword set — these span every domain and carry no
# inspiration value. CJK text rarely produces cross-domain function words
# as single tokens, so an English-only list suffices here.
_EN_STOPWORDS = frozenset("""
the and of to in for is on that with we this are as by be it from at or
an was were has have had not but they their which will can its our these
such then than there been more also may would could should into other
over under between through during before after above below both each few
all any most some own same so no nor only very just one two
using used use based et al
""".split())


_URL_FRAGMENTS = frozenset(
    {"https", "http", ".org", ".com", ".edu", "www", "url", "arxiv", "doi", "html"}
)


def _is_inspiring_token(text: str) -> bool:
    """Meaningful, not a stopword, not a fragment — the bar for the Aura.

    ``text`` is the RAW decode (may include a leading space). BPE property:
    English tokens that begin a word decode WITH a leading space; mid-word
    fragments (``ing``, ``ization``, ``Koop``) decode without one. We use
    this to reject ASCII fragments cheaply.
    """
    raw = text
    stripped = raw.strip()  # tokenizer.decode may include leading spaces
    if not _is_meaningful_text(stripped):
        return False
    if stripped.lower() in _EN_STOPWORDS:
        return False
    if stripped.lower() in _URL_FRAGMENTS:
        return False
    # pure digits (years "202", sizes "100") carry no inspiration
    if stripped.isdigit():
        return False
    # ASCII word: require the leading-space signature of a whole word
    is_ascii_word = all(ord(ch) < 128 for ch in stripped)
    if is_ascii_word and len(stripped) >= 3 and not raw.startswith(" "):
        return False  # mid-word BPE fragment like "ing" / "ization"
    # short ASCII fragments like "al", "et" — require ≥3 chars unless CJK
    if len(stripped) < 3 and not any("\u4e00" <= ch <= "\u9fff" for ch in stripped):
        return False
    return True


def compute_aura_tokens(
    clusters: list[dict],
    token_details: list[list[dict]] | None = None,
    centroids: list[dict[int, float]] | None = None,
    decode=None,
    min_clusters: int = 2,
    top_k: int = 40,
) -> list[dict]:
    """Compute cross-boundary (Aura) tokens from cluster centroids.

    Two input modes:
    1. ``centroids`` + ``decode`` (preferred): full centroid dicts and a
       ``decode(token_id) -> text`` callable. Uses ALL centroid tokens —
       cross-boundary signal often lives below the top-20.
    2. ``token_details`` (legacy): decoded top-token lists per cluster,
       each entry ``{"token_id", "weight", "decoded"}``. Limited to the
       top-K already decoded by the caller.

    Parameters
    ----------
    clusters : list of cluster dicts (needs ``cluster_id``)
    min_clusters : a token must appear in at least this many clusters
    top_k : keep at most this many aura tokens (by weight)
    decode : callable(token_id) -> str, required with ``centroids``

    Returns
    -------
    Sorted list of dicts:
      {"text", "weight", "clusterIds", "strengths", "entropy", "total"}
    """
    token_map: dict[int, dict[str, float]] = {}
    decoded_map: dict[int, str] = {}

    if centroids is not None:
        if decode is None:
            raise ValueError("decode callable is required when centroids are given")
        for cluster, centroid in zip(clusters, centroids):
            cid = cluster["cluster_id"]
            for tid, w in centroid.items():
                token_map.setdefault(int(tid), {})[cid] = float(w)
        for tid in token_map:
            try:
                decoded_map[tid] = decode(tid)
            except Exception:
                decoded_map[tid] = ""
    elif token_details is not None:
        for cluster, details in zip(clusters, token_details):
            cid = cluster["cluster_id"]
            for t in details:
                tid = t["token_id"]
                token_map.setdefault(tid, {})[cid] = t["weight"]
                decoded_map[tid] = t["decoded"]
    else:
        raise ValueError("provide either centroids+decode or token_details")

    aura: list[dict] = []
    for tid, dist in token_map.items():
        if len(dist) < min_clusters:
            continue  # single-cluster token: zero aura, sinks into planet

        text = decoded_map.get(tid, "")
        if not _is_inspiring_token(text):
            continue

        total = sum(dist.values())
        if total <= 0:
            continue

        q = [w / total for w in dist.values()]
        entropy = -sum(qi * math.log(qi) for qi in q if qi > 0)

        aura.append(
            {
                "tokenId": tid,
                "text": text.strip().replace("\n", " "),
                "clusterIds": list(dist.keys()),
                "strengths": {cid: round(w, 4) for cid, w in dist.items()},
                "entropy": round(entropy, 4),
                "total": round(total, 4),
                "weight": round(total * entropy, 4),
            }
        )

    aura.sort(key=lambda x: -x["weight"])
    return aura[:top_k]


def aura_summary(aura_tokens: list[dict], clusters: list[dict]) -> dict:
    """Human-readable summary of the aura layer for logging."""
    if not aura_tokens:
        return {"aura_tokens": 0, "bridged_pairs": []}

    pairs: set[frozenset[str]] = set()
    for at in aura_tokens:
        cids = at["clusterIds"]
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                pairs.add(frozenset((cids[i], cids[j])))

    label_of = {c["cluster_id"]: (c.get("label") or c["cluster_id"]) for c in clusters}
    bridged = []
    for p in pairs:
        items = list(p)
        if len(items) == 2:
            bridged.append(f"{label_of.get(items[0], items[0])} ↔ {label_of.get(items[1], items[1])}")

    return {
        "aura_tokens": len(aura_tokens),
        "bridged_pairs": sorted(bridged),
    }

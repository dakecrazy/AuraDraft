"""TokenClusterEngine — self-organising token-frequency clustering.

M2: pure statistics, no LLM.  Each cluster is a sparse centroid vector
(L2-normalised token frequencies).  New documents are classified by
cosine similarity to existing centroids; if no centroid is close enough
a new cluster is created.

The engine is **stateless across restarts**: clusters are loaded from
the ClusterStore on init and saved after every mutation.
"""

from __future__ import annotations

import math
import uuid
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from kb_agent.cluster.model import KnowledgeCluster

if TYPE_CHECKING:
    from kb_agent.storage.cluster_store import ClusterStore
    from kb_agent.tokenizer.canonical import CanonicalTokenizer


class TokenClusterEngine:
    """Self-organising token-frequency clustering engine.

    Parameters
    ----------
    tokenizer : CanonicalTokenizer
        The canonical tokenizer for encoding text.
    store : ClusterStore
        Persistence layer for clusters and token doc-frequency.
    similarity_threshold : float
        Minimum cosine similarity to assign a document to an existing
        cluster (default 0.35).  Below this → new cluster created.
    signature_top_k : int
        Number of top-weighted tokens to keep in the signature vector
        (default 128).  Higher = more precise but slower.
    centroid_prune_threshold : float
        Centroid weights below this value are dropped after each update
        (default 0.001).  Keeps centroids sparse.
    """

    def __init__(
        self,
        tokenizer: CanonicalTokenizer,
        store: ClusterStore,
        similarity_threshold: float = 0.35,
        signature_top_k: int = 128,
        centroid_prune_threshold: float = 0.001,
    ):
        self.tokenizer = tokenizer
        self.store = store
        self.threshold = similarity_threshold
        self.signature_top_k = signature_top_k
        self.prune = centroid_prune_threshold

        # In-memory cluster dict: cluster_id → KnowledgeCluster
        self.clusters: dict[str, KnowledgeCluster] = {}
        self._load_clusters()

        # Cache for meaningful-token check (avoids repeated decode)
        self._meaningful_cache: dict[int, bool] = {}

    # ── noise filter ──────────────────────────────────────────────

    def _is_meaningful_token(self, token_id: int) -> bool:
        """Check if a token decodes to meaningful text (not punctuation/whitespace/byte-fragment)."""
        if token_id in self._meaningful_cache:
            return self._meaningful_cache[token_id]
        decoded = self.tokenizer.decode([token_id])
        result = self._is_meaningful_text(decoded)
        self._meaningful_cache[token_id] = result
        return result

    @staticmethod
    def _is_meaningful_text(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if "\ufffd" in text:
            return False
        # Reject single characters — they're stopwords in any language
        if len(stripped) < 2:
            return False
        # Reject if no Letter or Number category characters
        import unicodedata
        has_alnum = any(
            unicodedata.category(ch)[0] in ("L", "N") for ch in stripped
        )
        return has_alnum

    # ── public API ────────────────────────────────────────────────

    def classify_and_assign(
        self,
        doc_id: str,
        text: str,
        method: str = "tfidf_topk",
    ) -> dict:
        """Classify a document and assign it to a cluster.

        Steps:
          1. Extract token signature (TF-IDF or TF)
          2. Compare to all existing centroids
          3. Assign to best-matching cluster OR create a new one
          4. Update token doc-frequency (AFTER classification, before
             centroid update, so the doc doesn't "see itself" in IDF)

        Returns a dict with action, cluster_id, cluster_label, similarity.
        """
        # Step 1: extract signature (uses IDF *before* this doc)
        signature = self._extract_signature(text, method)

        if not self.clusters:
            # First document → create first cluster
            result = self._create_new_cluster(doc_id, signature, text)
            self._update_doc_freq(doc_id)
            return result

        # Step 2: compare to all centroids
        best_cid, best_sim = self._find_best_match(signature)

        # Step 3: assign or create
        if best_sim >= self.threshold:
            self._add_to_cluster(best_cid, doc_id, signature)
            self._update_doc_freq(doc_id)
            return {
                "action": "assigned",
                "cluster_id": best_cid,
                "cluster_label": self.clusters[best_cid].label,
                "similarity": round(best_sim, 4),
            }
        else:
            result = self._create_new_cluster(doc_id, signature, text)
            self._update_doc_freq(doc_id)
            return result

    def get_cluster_count(self) -> int:
        return len(self.clusters)

    def get_all_clusters(self) -> list[KnowledgeCluster]:
        return list(self.clusters.values())

    def get_cluster(self, cluster_id: str) -> KnowledgeCluster | None:
        return self.clusters.get(cluster_id)

    # ── debug ─────────────────────────────────────────────────────

    def debug_signature(self, text: str, top_k: int = 20) -> dict:
        """Print top-K signature tokens with their decoded text for diagnosis."""
        sig = self._extract_signature(text, method="tfidf_topk")
        top = sorted(sig.items(), key=lambda x: -x[1])[:top_k]
        rows = []
        for tid, weight in top:
            decoded = self.tokenizer.decode([tid])
            rows.append(f"  ID={tid:6d}  weight={weight:.4f}  decoded={decoded!r}")
        return {"top_tokens": rows, "total_unique": len(sig)}

    def _extract_signature(
        self,
        text: str,
        method: str = "tfidf_topk",
    ) -> dict[int, float]:
        """Extract a sparse L2-normalised token signature from *text*.

        Methods:
          "tf_topk"    — raw term frequency, top-K
          "tfidf_topk" — TF-IDF weighted, top-K (default)
        """
        token_ids = self.tokenizer.encode(text)

        if method == "tf_topk":
            return self._tf_signature(token_ids)
        else:
            return self._tfidf_signature(token_ids)

    def _tf_signature(self, token_ids: list[int]) -> dict[int, float]:
        """Raw term frequency → L2-normalised top-K."""
        freq: dict[int, int] = defaultdict(int)
        for tid in token_ids:
            if not self._is_meaningful_token(tid):
                continue
            freq[tid] += 1

        # Top-K by frequency
        sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])[
            : self.signature_top_k
        ]

        # L2 normalise
        norm = math.sqrt(sum(f**2 for _, f in sorted_tokens))
        if norm == 0:
            return {}
        return {tid: f / norm for tid, f in sorted_tokens}

    def _tfidf_signature(self, token_ids: list[int]) -> dict[int, float]:
        """TF-IDF weighted → L2-normalised top-K.

        When N < 50, IDF has no statistical meaning — falls back to TF-only.
        """
        freq: dict[int, int] = defaultdict(int)
        for tid in token_ids:
            if not self._is_meaningful_token(tid):
                continue
            freq[tid] += 1

        total_tokens = len(token_ids)
        N = self.store.get_total_docs()

        # For small corpora, skip IDF — it's statistically meaningless
        if N < 50:
            return self._tf_signature(token_ids)

        tfidf: dict[int, float] = {}
        for tid, tf in freq.items():
            df = self.store.get_doc_freq(tid)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0) if N > 0 else 1.0
            tfidf[tid] = (tf / total_tokens) * idf

        # Top-K by TF-IDF weight
        sorted_tokens = sorted(tfidf.items(), key=lambda x: -x[1])[
            : self.signature_top_k
        ]

        norm = math.sqrt(sum(v**2 for _, v in sorted_tokens))
        if norm == 0:
            return {}
        return {tid: v / norm for tid, v in sorted_tokens}

    # ── internal: cosine similarity ───────────────────────────────

    @staticmethod
    def _sparse_cosine(
        vec_a: dict[int, float],
        vec_b: dict[int, float],
    ) -> float:
        """Cosine similarity between two sparse dict vectors."""
        common = set(vec_a.keys()) & set(vec_b.keys())
        if not common:
            return 0.0
        dot = sum(vec_a[k] * vec_b[k] for k in common)
        norm_a = math.sqrt(sum(v**2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v**2 for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _find_best_match(
        self, signature: dict[int, float]
    ) -> tuple[str | None, float]:
        """Find the cluster with the highest cosine similarity to *signature*.

        Returns ``(cluster_id, similarity)``.  If no clusters exist,
        returns ``(None, 0.0)``.
        """
        if not self.clusters:
            return None, 0.0
        best_cid: str | None = None
        best_sim = -1.0
        for cid, cluster in self.clusters.items():
            sim = self._sparse_cosine(signature, cluster.centroid)
            if sim > best_sim:
                best_sim = sim
                best_cid = cid
        return best_cid, best_sim

    # ── internal: cluster mutation ────────────────────────────────

    def _add_to_cluster(
        self,
        cluster_id: str,
        doc_id: str,
        signature: dict[int, float],
    ) -> None:
        """Add a document to an existing cluster (running-average centroid)."""
        cluster = self.clusters[cluster_id]
        n = cluster.doc_count

        # Running average: new_centroid = (old * n + sig) / (n + 1)
        new_centroid: dict[int, float] = {}
        all_keys = set(cluster.centroid.keys()) | set(signature.keys())
        for k in all_keys:
            old_val = cluster.centroid.get(k, 0.0)
            new_val = signature.get(k, 0.0)
            new_centroid[k] = (old_val * n + new_val) / (n + 1)

        # Prune low-weight tokens
        new_centroid = {k: v for k, v in new_centroid.items() if v > self.prune}

        cluster.centroid = new_centroid
        cluster.member_doc_ids.append(doc_id)
        cluster.doc_count = n + 1
        cluster.last_updated = KnowledgeCluster.now()

        # Persist
        self.store.save_cluster(cluster)

    def _create_new_cluster(
        self,
        doc_id: str,
        signature: dict[int, float],
        text: str,
    ) -> dict:
        """Create a new cluster seeded with *signature* as its centroid."""
        cluster_id = uuid.uuid4().hex[:8]
        now = KnowledgeCluster.now()

        # Generate a fallback label from the top-K decoded tokens
        label = self._generate_label(signature)

        cluster = KnowledgeCluster(
            cluster_id=cluster_id,
            centroid=signature.copy(),
            member_doc_ids=[doc_id],
            label=label,
            doc_count=1,
            created_at=now,
            last_updated=now,
        )
        self.clusters[cluster_id] = cluster
        self.store.save_cluster(cluster)

        return {
            "action": "new_cluster_created",
            "cluster_id": cluster_id,
            "cluster_label": label,
            "similarity": 0.0,
        }

    def _generate_label(self, signature: dict[int, float]) -> str:
        """Generate a human-readable label from the centroid's top tokens.

        M2 fallback: decode the top-5 token IDs.  For BPE tokenizers
        (especially cl100k_base on CJK text) individual token IDs may
        be byte-level fragments that don't form valid UTF-8, so we
        fall back to a generic placeholder.
        """
        top_tokens = sorted(signature.items(), key=lambda x: -x[1])[:5]
        top_ids = [tid for tid, _ in top_tokens]
        raw = self.tokenizer.decode(top_ids)
        # Strip whitespace / newlines / replacement characters
        label = raw.strip().replace("\n", " ").replace("\ufffd", "")[:40]
        label = label.strip()
        return label if label and len(label) >= 2 else f"topic_{uuid.uuid4().hex[:4]}"

    # ── internal: persistence ─────────────────────────────────────

    def _load_clusters(self) -> None:
        """Load all clusters from the store into memory."""
        for cluster in self.store.load_all_clusters():
            self.clusters[cluster.cluster_id] = cluster

    def _update_doc_freq(self, doc_id: str) -> None:
        """Update token document frequency for a document by its doc_id.

        Loads text from the ``documents`` table internally.
        """
        from kb_agent.document.loader import load_text
        conn = self.store.db.connect()
        row = conn.execute(
            "SELECT file_path FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not row:
            return
        text = load_text(row["file_path"])
        token_ids = self.tokenizer.encode(text)
        self.store.update_doc_freq(token_ids)
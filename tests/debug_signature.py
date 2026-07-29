"""Diagnostic: inspect TF-IDF signature tokens for all 5 fixtures."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kb_agent.tokenizer.canonical import CanonicalTokenizer
from kb_agent.storage.db import Database
from kb_agent.storage.cluster_store import ClusterStore
from kb_agent.cluster.manager import TokenClusterEngine
from kb_agent.document.loader import load_text

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sample_docs"
FILES = [
    ("dl_001", "deep_learning_attention.txt"),
    ("legal_001", "legal_contract.txt"),
    ("quantum_001", "quantum_computing.txt"),
    ("cooking_001", "cooking_recipe.txt"),
    ("dl_002", "deep_learning_training.txt"),
]

import tempfile
db_path = tempfile.mktemp(suffix=".db")
db = Database(db_path)
tokenizer = CanonicalTokenizer()
store = ClusterStore(db)
engine = TokenClusterEngine(tokenizer, store, similarity_threshold=0.35)

for doc_id, filename in FILES:
    text = load_text(str(FIXTURES / filename))
    sig = engine.debug_signature(text, top_k=20)
    print(f"\n{'='*60}")
    print(f"  {doc_id} — {filename}")
    print(f"{'='*60}")
    print(f"  Total unique tokens in signature: {sig['total_unique']}")
    for line in sig['top_tokens']:
        print(line)

Path(db_path).unlink(missing_ok=True)
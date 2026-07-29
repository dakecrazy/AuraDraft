# DEPRECATED: 被 tools/cli.py 替代，保留仅为兼容
"""CLI entry point for kb-agent.

Usage:
  kb-agent ingest <file> [--category CAT] [--tags TAG ...]
  kb-agent query <question>
  kb-agent stats
  kb-agent clusters
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is on the path when run as script
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from kb_agent.tokenizer.canonical import CanonicalTokenizer
from kb_agent.storage.db import Database
from kb_agent.storage.cluster_store import ClusterStore
from kb_agent.index.engine import TokenIndexEngine
from kb_agent.cluster.manager import TokenClusterEngine
from kb_agent.llm.client import MockLLMClient
from kb_agent.router.moe_router import MoERouter
from kb_agent.pipeline.ingest import IngestionPipeline
from kb_agent.pipeline.query import QueryPipeline


def build_pipeline(db_path: str = "kb_index.db"):
    """Build the full dependency chain."""
    db = Database(db_path)
    tokenizer = CanonicalTokenizer()
    store = ClusterStore(db)
    index_engine = TokenIndexEngine(db, tokenizer, chunk_size=256, chunk_overlap=32)
    cluster_engine = TokenClusterEngine(
        tokenizer, store, similarity_threshold=0.35
    )
    llm = MockLLMClient()
    router = MoERouter(cluster_engine, llm, top_k_candidates=5, min_similarity=0.05)
    ingest_pipeline = IngestionPipeline(
        index_engine, cluster_engine, router, llm, store
    )
    query_pipeline = QueryPipeline(index_engine, cluster_engine, llm)
    return ingest_pipeline, query_pipeline, cluster_engine


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest one or more documents into the knowledge base."""
    path = Path(args.file)
    if not path.exists():
        print(f"❌ Path not found: {args.file}")
        sys.exit(1)

    pipeline, _, _ = build_pipeline(args.db)

    if path.is_dir():
        # Batch ingest: recursively find all supported documents
        from kb_agent.document.loader import iter_documents
        files = iter_documents(str(path))
        if not files:
            print(f"⚠️  No supported documents found in {args.file}")
            print(f"   Supported: .txt .md .rst .log")
            return
        print(f"📂 批量摄入 {len(files)} 个文档 from {args.file}")
        for f in files:
            result = pipeline.ingest(
                file_path=str(f),
                category=args.category or "",
                tags=args.tags or [],
            )
            cls = result["classification"]
            print(f"  {f.name:40s} → {cls['action']:22s} {cls.get('cluster_label', '')[:20]}")
    else:
        # Single file ingest
        result = pipeline.ingest(
            file_path=args.file,
            category=args.category or "",
            tags=args.tags or [],
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_query(args: argparse.Namespace) -> None:
    """Query the knowledge base."""
    _, pipeline, _ = build_pipeline(args.db)
    result = pipeline.query(args.question, mode=args.mode)
    print(f"\n{'='*60}")
    print(f"问题: {args.question}")
    print(f"{'='*60}")
    print(f"\n回答:\n{result['answer']}")
    if result["sources"]:
        print(f"\n来源 ({len(result['sources'])} 篇):")
        for s in result["sources"]:
            print(f"  [{s['score']:.4f}] {s['file_path']} ({s['category']})")
    if result["relevant_clusters"]:
        print(f"\n相关领域:")
        for label, sim in result["relevant_clusters"].items():
            print(f"  {label}: {sim:.3f}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Show knowledge base statistics."""
    pipeline, query_pipeline, cluster_engine = build_pipeline(args.db)
    index_stats = pipeline.index.get_stats()

    print(f"\n{'='*60}")
    print("知识库统计")
    print(f"{'='*60}")
    print(f"  文档总数: {index_stats['total_documents']}")
    print(f"  索引 Token 数: {index_stats['total_tokens_indexed']}")
    print(f"  唯一 Token 类型: {index_stats['unique_token_types']}")
    print(f"  平均文档长度: {index_stats['avg_document_length']} tokens")
    print(f"  簇数量: {cluster_engine.get_cluster_count()}")
    print()
    for c in cluster_engine.get_all_clusters():
        print(f"  📂 {c.label} ({c.cluster_id[:8]}): {c.doc_count} 篇文档")
        if c.knowledge_card:
            print(f"     知识档案: {c.knowledge_card[:80]}...")


def cmd_clusters(args: argparse.Namespace) -> None:
    """List all clusters with details."""
    _, _, cluster_engine = build_pipeline(args.db)
    print(f"\n{'='*60}")
    print(f"知识簇列表 (共 {cluster_engine.get_cluster_count()} 个)")
    print(f"{'='*60}")
    for c in cluster_engine.get_all_clusters():
        print(f"\n📂 {c.label} ({c.cluster_id})")
        print(f"  文档数: {c.doc_count}")
        print(f"  质心 Token 数: {len(c.centroid)}")
        if c.knowledge_card:
            print(f"  知识档案:")
            for line in c.knowledge_card.strip().split("\n")[:10]:
                print(f"    {line}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="kb-agent: Token-based knowledge base with MoE routing"
    )
    parser.add_argument(
        "--db",
        default="kb_index.db",
        help="SQLite database path (default: kb_index.db)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a document")
    p_ingest.add_argument("file", help="Path to the document file")
    p_ingest.add_argument("--category", "-c", help="Document category")
    p_ingest.add_argument("--tags", "-t", nargs="*", help="Document tags")
    p_ingest.set_defaults(func=cmd_ingest)

    # query
    p_query = sub.add_parser("query", help="Query the knowledge base")
    p_query.add_argument("question", help="Natural language question")
    p_query.add_argument(
        "--mode",
        choices=["exact", "phrase", "hybrid"],
        default="hybrid",
        help="Search mode (default: hybrid)",
    )
    p_query.set_defaults(func=cmd_query)

    # stats
    p_stats = sub.add_parser("stats", help="Show knowledge base statistics")
    p_stats.set_defaults(func=cmd_stats)

    # clusters
    p_clusters = sub.add_parser("clusters", help="List all clusters")
    p_clusters.set_defaults(func=cmd_clusters)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
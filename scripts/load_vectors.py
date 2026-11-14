"""Rebuild the Day 6 vector index (strategy B) from data/chunks.json into an empty or existing pgvector database.

Why: graders do not have the author's Day 6 database. This script recreates the same index from the shared chunk
file with the same embedding model (text-embedding-3-small, 1536 dims) and the Day 6 `PgVectorStore` schema.

  python -X utf8 scripts/load_vectors.py --dry-run     # chunk/char stats and cost estimate, no API call, no DB write
  python -X utf8 scripts/load_vectors.py               # embed + upsert (about 0.01 USD for 1,310 chunks)

Needs DATABASE_URL and OPENAI_API_KEY (in .env or the environment). Re-running is safe: rows are upserted by
(strategy, chunk_id), so the count never doubles, and obsolete chunks of the same documents are removed.
The script fails loudly on any API/DB error; it never falls back to fake vectors.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOKENS_PER_CHAR = 0.54            # measured on Day 6: 1,176,944 tokens for 2,199,180 chars (Vietnamese text)
USD_PER_1M = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}   # published price, not re-verified


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=Path, default=ROOT / "data" / "chunks.json")
    p.add_argument("--strategy", default="B", help="label stored in the `strategy` column (retrievers read 'B')")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    load_dotenv(ROOT / ".env")

    rows = json.loads(args.chunks.read_text(encoding="utf-8"))
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate chunk id in input")
    chars = sum(len(r["text"]) for r in rows)
    model = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
    est_tokens = int(chars * TOKENS_PER_CHAR)
    est_cost = est_tokens * USD_PER_1M.get(model, 0) / 1_000_000
    print(f"{len(rows)} chunks, {chars:,} chars, model={model}, est. {est_tokens:,} tokens ~ ${est_cost:.4f}")
    if args.dry_run:
        return

    from src.day6.embeddings import MyEmbedder
    from src.day6.models import Chunk
    from src.day6.store import PgVectorStore

    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set (see .env.example)")
    embedder = MyEmbedder(model)
    store = PgVectorStore(url, embedder.dimensions)
    store.create_schema()                       # also fails if an old table has another vector size
    before = store.count(args.strategy)
    chunks = [Chunk(chunk_id=r["id"], document_id=r["source_id"], source=r["source"], page=r["page"],
                    text=r["text"], metadata=r["metadata"]) for r in rows]
    t0 = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in chunks])
    embed_s = time.perf_counter() - t0
    store.upsert(args.strategy, chunks, vectors)
    after = store.count(args.strategy)
    print(json.dumps({"strategy": args.strategy, "rows_before": before, "rows_after": after, "expected": len(rows),
                      "dimensions": embedder.dimensions, "embedding_tokens": embedder.tokens_used,
                      "requests": embedder.requests, "embed_seconds": round(embed_s, 1),
                      "cost_usd_estimate": round(embedder.tokens_used * USD_PER_1M.get(model, 0) / 1_000_000, 5)}, indent=1))
    if after != len(rows):
        raise SystemExit(f"row count {after} != {len(rows)} chunks: index is inconsistent")
    print("OK: index rebuilt")


if __name__ == "__main__":
    main()

"""Dense retrieval over the Day 6 pgvector store (strategy B, text-embedding-3-small).

Returns chunks in the Day 7 schema: id, text, source_id, source, page, metadata, score, rank.
Errors (missing key, DB down) propagate; there is no mock fallback.
"""
import os
import time
from typing import Any

from dotenv import load_dotenv

from src.day6.embeddings import MyEmbedder
from src.day6.store import PgVectorStore

STRATEGY = "B"


class DenseRetriever:
    name = "dense"

    def __init__(self, strategy: str = STRATEGY) -> None:
        load_dotenv()
        self.strategy = strategy
        self.embedder = MyEmbedder(os.getenv("EMBEDDING_MODEL") or None)
        self.store = PgVectorStore(os.environ["DATABASE_URL"], self.embedder.dimensions)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        vec = self.embedder.embed_query(query)
        rows = self.store.search(self.strategy, vec, top_k=top_k, filters=filters)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return [{"id": r["chunk_id"], "text": r["text"], "source_id": r["document_id"], "source": r["source"],
                 "page": r["page"], "metadata": r["metadata"], "score": r["score"], "rank": i + 1,
                 "method": self.name} for i, r in enumerate(rows)]

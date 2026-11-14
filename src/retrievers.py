"""Retrievers over the same Day 6 chunk set (strategy B, 1310 chunks).

All retrievers share one interface and return the Day 7 chunk schema plus
`score`, `rank` (1-based) and `method`:

    search(query, *, filters=None, top_k=3) -> list[dict]

- DenseRetriever  : src/dense.py (pgvector, text-embedding-3-small)  [Day 6 index]
- Bm25Retriever   : lexical BM25 over data/chunks.json (no external service)
- HybridRetriever : Reciprocal Rank Fusion of dense + BM25 candidates

`filters` is only ever supplied by the caller (user context), never by the model
or by benchmark gold. Supported key: any metadata field, e.g. {"period": "truoc-cach-mang"}.
"""
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "chunks.json"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text).replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", "".join(ch for ch in text if unicodedata.category(ch) != "Mn"))


def tokenize(text: str, mode: str = "syllable", fold: bool = False) -> list[str]:
    """Vietnamese tokens. Words are space-separated syllables, so:
    - syllable : one token per syllable
    - bigram   : adjacent syllable pairs (approximates compound words)
    - both     : syllables + bigrams
    `fold=True` removes diacritics (helps typos/no-accent queries, hurts disambiguation)."""
    text = unicodedata.normalize("NFC", text).casefold()
    if fold:
        text = strip_accents(text)
    syl = re.findall(r"\w+", text)
    bi = [f"{a}_{b}" for a, b in zip(syl, syl[1:])]
    return {"syllable": syl, "bigram": bi, "both": syl + bi}[mode]


def _match(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    return all(chunk.get(k, chunk["metadata"].get(k)) == v for k, v in (filters or {}).items())


def _row(chunk: dict[str, Any], score: float, rank: int, method: str) -> dict[str, Any]:
    return {"id": chunk["id"], "text": chunk["text"], "source_id": chunk["source_id"], "source": chunk["source"],
            "page": chunk["page"], "metadata": chunk["metadata"], "score": float(score), "rank": rank, "method": method}


class Bm25Retriever:
    """Okapi BM25 (k1, b configurable). Chunks with score 0 are never returned."""

    def __init__(self, chunks: list[dict[str, Any]] | None = None, *, mode: str = "syllable",
                 fold: bool = False, k1: float = 1.2, b: float = 0.75) -> None:
        self.chunks = chunks if chunks is not None else load_chunks()
        self.mode, self.fold, self.k1, self.b = mode, fold, k1, b
        self.name = "bm25"
        docs = [Counter(tokenize(c["text"], mode, fold)) for c in self.chunks]
        self.lengths = [sum(d.values()) for d in docs]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        df = Counter(t for d in docs for t in d)
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            for t, tf in d.items():
                self.postings.setdefault(t, []).append((i, tf))

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        scores: dict[int, float] = {}
        for t in set(tokenize(query, self.mode, self.fold)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in self.postings[t]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / norm
        ranked = sorted((i for i in scores if _match(self.chunks[i], filters)),
                        key=lambda i: (-scores[i], self.chunks[i]["id"]))[:top_k]
        return [_row(self.chunks[i], scores[i], r, self.name) for r, i in enumerate(ranked, 1)]


def rrf(rankings: list[list[dict[str, Any]]], *, k: int = 60) -> list[tuple[dict[str, Any], float]]:
    """Reciprocal Rank Fusion: score(d) = sum_over_lists 1/(k + rank). Rank-based, so no
    score normalization is needed between cosine similarity and BM25."""
    fused: dict[str, float] = {}
    first: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        for r, row in enumerate(ranking, 1):
            fused[row["id"]] = fused.get(row["id"], 0.0) + 1.0 / (k + r)
            first.setdefault(row["id"], row)
    return sorted(((first[i], s) for i, s in fused.items()), key=lambda x: (-x[1], x[0]["id"]))


class HybridRetriever:
    """Dense + BM25 candidates (each `pool` deep) fused with RRF."""

    def __init__(self, dense, lexical, *, pool: int = 20, k: int = 60) -> None:
        self.dense, self.lexical, self.pool, self.k = dense, lexical, pool, k
        self.name = "hybrid"

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        d = self.dense.search(query, filters=filters, top_k=self.pool)
        l = self.lexical.search(query, filters=filters, top_k=self.pool)
        fused = rrf([d, l], k=self.k)[:top_k]
        return [{**{k_: v for k_, v in row.items() if k_ not in ("score", "rank", "method")},
                 "score": s, "rank": r, "method": self.name} for r, (row, s) in enumerate(fused, 1)]

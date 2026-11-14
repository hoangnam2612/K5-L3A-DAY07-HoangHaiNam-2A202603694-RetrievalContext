"""TODO: store and search vectors using PostgreSQL + pgvector."""
import json
from typing import Any

import psycopg

from .models import Chunk

# Implementation notes (added): one table `chunks` for all strategies, keyed by (strategy, chunk_id).
# Table/column names are fixed in code; only VALUES are parameterized, and filter fields are
# restricted to an allow-list, so user-supplied filter input never becomes SQL text.
_FILTER_COLUMNS = {"chapter", "chapter_title", "section", "page", "document_id"}
# Fields kept only in the JSONB `metadata` column (Markdown corpus): filtered with metadata ->> key.
_FILTER_METADATA = {"period", "unit_id", "title", "file"}


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        """TODO: retain connection settings; validate model dimensions."""
        if not database_url:
            raise ValueError("database_url is empty (set DATABASE_URL in starter/.env)")
        if not isinstance(dimensions, int) or not 0 < dimensions <= 2000:
            raise ValueError(f"dimensions must be an int in 1..2000, got {dimensions!r}")
        self.database_url = database_url
        self.dimensions = dimensions

    def _connect(self):
        return psycopg.connect(self.database_url)

    @staticmethod
    def _vec(v: list[float]) -> str:
        return "[" + ",".join(repr(float(x)) for x in v) + "]"

    def create_schema(self) -> None:
        """TODO: create table with vector column and source/page/filter metadata."""
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    strategy      TEXT    NOT NULL,
                    chunk_id      TEXT    NOT NULL,
                    document_id   TEXT    NOT NULL,
                    source        TEXT    NOT NULL,
                    page          INTEGER NOT NULL,
                    chapter       INTEGER,
                    chapter_title TEXT,
                    section       TEXT,
                    text          TEXT    NOT NULL,
                    metadata      JSONB   NOT NULL DEFAULT '{{}}',
                    embedding     vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )""")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_filter_idx ON chunks (strategy, chapter)")
            # Fail loudly if an old table was built with a different embedding size.
            row = conn.execute("""SELECT atttypmod FROM pg_attribute
                                  WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'""").fetchone()
            if row and row[0] != self.dimensions:
                raise RuntimeError(f"table chunks has vector({row[0]}), but this store uses {self.dimensions}; "
                                   "drop the table or re-embed with the same model")

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """TODO: validate lengths/dimensions and write stable IDs without duplicates.

        When re-ingesting changed source documents, remove obsolete chunks too.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        ids = [c.chunk_id for c in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate chunk_id in input")
        for c, v in zip(chunks, vectors):
            if len(v) != self.dimensions:
                raise ValueError(f"{c.chunk_id}: vector has {len(v)} dims, expected {self.dimensions}")
        sql = """
            INSERT INTO chunks (strategy, chunk_id, document_id, source, page, chapter, chapter_title,
                                section, text, metadata, embedding)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
            ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                document_id=EXCLUDED.document_id, source=EXCLUDED.source, page=EXCLUDED.page,
                chapter=EXCLUDED.chapter, chapter_title=EXCLUDED.chapter_title, section=EXCLUDED.section,
                text=EXCLUDED.text, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding"""
        rows = [(strategy, c.chunk_id, c.document_id, c.source, c.page, c.metadata.get("chapter"),
                 c.metadata.get("chapter_title"), c.metadata.get("section"), c.text,
                 json.dumps(c.metadata, ensure_ascii=False), self._vec(v))
                for c, v in zip(chunks, vectors)]
        docs = sorted({c.document_id for c in chunks})
        with self._connect() as conn:  # one transaction: all-or-nothing
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
                # remove obsolete chunks of the re-ingested documents (same strategy only)
                cur.execute("DELETE FROM chunks WHERE strategy = %s AND document_id = ANY(%s) "
                            "AND NOT (chunk_id = ANY(%s))", (strategy, docs, ids))

    def count(self, strategy: str) -> int:
        """Number of stored chunks for a strategy (used to prove re-ingestion does not duplicate)."""
        with self._connect() as conn:
            return conn.execute("SELECT count(*) FROM chunks WHERE strategy = %s", (strategy,)).fetchone()[0]

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """TODO: filter permitted metadata in SQL, rank by cosine distance, return top-k.

        Return chunk ID, text, score and metadata. Parameterize user-supplied values.
        """
        if len(query_vector) != self.dimensions:
            raise ValueError(f"query vector has {len(query_vector)} dims, expected {self.dimensions}")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        where, params = ["strategy = %s"], [strategy]
        for key, value in (filters or {}).items():
            if key in _FILTER_COLUMNS:
                where.append(f"{key} = %s")
                params.append(value)
            elif key in _FILTER_METADATA:
                where.append("metadata ->> %s = %s")   # both the key and the value are bound parameters
                params.extend([key, str(value)])
            else:
                raise ValueError(f"filter field {key!r} not allowed; use one of "
                                 f"{sorted(_FILTER_COLUMNS | _FILTER_METADATA)}")
        qv = self._vec(query_vector)
        sql = f"""SELECT chunk_id, text, 1 - (embedding <=> %s::vector) AS score, document_id, source, page, metadata
                  FROM chunks WHERE {' AND '.join(where)}
                  ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s"""
        with self._connect() as conn:
            rows = conn.execute(sql, [qv, *params, qv, top_k]).fetchall()
        return [{"chunk_id": r[0], "text": r[1], "score": float(r[2]), "document_id": r[3],
                 "source": r[4], "page": r[5], "metadata": r[6]} for r in rows]

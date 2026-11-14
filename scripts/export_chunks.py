"""Export Day 6 strategy-B chunks from pgvector to data/chunks.json (Day 7 schema).

Usage: python -I -X utf8 scripts/export_chunks.py   (needs DATABASE_URL in .env or env)
"""
import json
import os
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
url = os.environ.get("DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab")
with psycopg.connect(url) as conn:
    rows = conn.execute("SELECT chunk_id, text, document_id, source, page, metadata "
                        "FROM chunks WHERE strategy = 'B' ORDER BY page, chunk_id").fetchall()
chunks = [{"id": r[0], "text": r[1], "source_id": r[2], "source": r[3], "page": r[4], "metadata": r[5]}
          for r in rows]
out = ROOT / "data" / "chunks.json"
out.write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"wrote {len(chunks)} chunks -> {out}")

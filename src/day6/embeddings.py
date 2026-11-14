"""Choose a real embedding provider (API or local) and implement this contract."""
import os
import time
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI

# Implementation notes (added): provider = OpenAI embeddings API; key and model come from starter/.env.
_KNOWN_DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
DEFAULT_MODEL = "text-embedding-3-small"
_BATCH = 96  # texts per request


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """TODO: implement a provider adapter; do not silently fall back to mock vectors."""

    def __init__(self, model_name: str | None = None) -> None:
        load_dotenv()
        model_name = model_name or os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing (set it in starter/.env); no mock fallback is used")
        self.model_name = model_name
        self._client = OpenAI()
        self.tokens_used = 0  # accumulated from API usage, for the experiment log / cost estimate
        self.requests = 0
        self.dimensions = _KNOWN_DIMENSIONS.get(model_name) or len(self._embed(["dimension probe"])[0])

    def _embed(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(5):  # simple retry with backoff for rate limits / transient errors
            try:
                resp = self._client.embeddings.create(model=self.model_name, input=texts)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        self.tokens_used += resp.usage.total_tokens
        self.requests += 1
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):
            out.extend(self._embed(texts[i:i + _BATCH]))
        return out

    def embed_query(self, question: str) -> list[float]:
        return self._embed([question])[0]

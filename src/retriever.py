"""Retrieve grounded context chunks for a user query."""

from __future__ import annotations

from dataclasses import dataclass

from src.embedder import embed_one
from src.ingest import Chunk
from src.vector_store import VectorStore

# Cosine similarity must exceed this to be considered grounded context.
GROUNDING_THRESHOLD = 0.35


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    scores: list[float]
    is_grounded: bool  # True when at least one chunk clears the threshold


def retrieve(query: str, store: VectorStore, top_k: int = 3) -> RetrievalResult:
    query_vec = embed_one(query)
    hits = store.search(query_vec, top_k=top_k)

    above = [(c, s) for c, s in hits if s >= GROUNDING_THRESHOLD]
    chunks = [c for c, _ in above]
    scores = [s for _, s in above]

    return RetrievalResult(
        chunks=chunks,
        scores=scores,
        is_grounded=len(above) > 0,
    )

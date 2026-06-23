"""In-memory vector store backed by NumPy cosine similarity."""

from __future__ import annotations

import numpy as np

from src.ingest import Chunk


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # shape (N, D)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Append chunks and their pre-computed (normalised) embeddings."""
        self._chunks.extend(chunks)
        self._matrix = (
            embeddings
            if self._matrix is None
            else np.vstack([self._matrix, embeddings])
        )

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """
        Return the top_k (chunk, cosine_score) pairs, sorted descending.
        query_vec must be L2-normalised (same convention as embedder.embed).
        """
        if self._matrix is None or len(self._chunks) == 0:
            return []

        scores: np.ndarray = self._matrix @ query_vec
        k = min(top_k, len(self._chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self._chunks[i], float(scores[i])) for i in top_indices]

    def __len__(self) -> int:
        return len(self._chunks)

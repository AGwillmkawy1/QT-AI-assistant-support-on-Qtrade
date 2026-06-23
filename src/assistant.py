"""
QTrade support assistant — orchestrates ingest, retrieval, escalation, and generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src import embedder, escalation, llm
from src.ingest import Chunk, load_docs
from src.retriever import RetrievalResult, retrieve
from src.vector_store import VectorStore


@dataclass
class AssistantResponse:
    answer: str
    sources: list[str]
    escalated: bool
    escalation_reason: escalation.EscalationReason
    retrieved_chunks: list[Chunk]


class Assistant:
    def __init__(self, docs_dir: str | Path = "docs") -> None:
        self._store = VectorStore()
        self._build_index(Path(docs_dir))

    def _build_index(self, docs_dir: Path) -> None:
        chunks = load_docs(docs_dir)
        if not chunks:
            raise ValueError(f"No .txt files found in {docs_dir}")
        texts = [c.text for c in chunks]
        embeddings = embedder.embed(texts)
        self._store.add(chunks, embeddings)

    def ask(self, query: str) -> AssistantResponse:
        result: RetrievalResult = retrieve(query, self._store, top_k=3)
        decision = escalation.check(query, result.is_grounded)

        if decision.should_escalate:
            return AssistantResponse(
                answer=decision.message,
                sources=[],
                escalated=True,
                escalation_reason=decision.reason,
                retrieved_chunks=result.chunks,
            )

        context = _format_context(result.chunks)
        answer = llm.generate(query, context)
        sources = _unique_sources(result.chunks)

        return AssistantResponse(
            answer=answer,
            sources=sources,
            escalated=False,
            escalation_reason=escalation.EscalationReason.NONE,
            retrieved_chunks=result.chunks,
        )


def _format_context(chunks: list[Chunk]) -> str:
    parts = []
    for chunk in chunks:
        parts.append(f"[{chunk.doc_name}]\n{chunk.text}")
    return "\n\n".join(parts)


def _unique_sources(chunks: list[Chunk]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for c in chunks:
        if c.doc_name not in seen:
            seen.add(c.doc_name)
            sources.append(c.doc_name)
    return sources

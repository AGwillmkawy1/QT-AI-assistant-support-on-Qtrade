"""Load and chunk help documents into overlapping text segments."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    text: str
    doc_name: str
    doc_path: str
    chunk_index: int


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries, keeping punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_document(path: Path, chunk_size: int = 3, overlap: int = 1) -> list[Chunk]:
    """
    Chunk a document by sliding a window over its sentences.

    chunk_size: sentences per chunk
    overlap:    sentences shared between consecutive chunks
    """
    raw = path.read_text(encoding="utf-8")

    # Strip the "Doc: <title>" header line so it doesn't pollute chunks,
    # but keep doc_name for citation.
    lines = raw.strip().splitlines()
    doc_name = lines[0].removeprefix("Doc:").strip() if lines[0].startswith("Doc:") else path.stem
    body = " ".join(lines[1:]).strip()

    sentences = _split_sentences(body)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(sentences), step):
        window = sentences[i : i + chunk_size]
        chunks.append(
            Chunk(
                text=" ".join(window),
                doc_name=doc_name,
                doc_path=str(path),
                chunk_index=len(chunks),
            )
        )
    return chunks


def load_docs(docs_dir: str | Path) -> list[Chunk]:
    """Load and chunk all .txt files in docs_dir."""
    docs_dir = Path(docs_dir)
    all_chunks: list[Chunk] = []
    for txt_file in sorted(docs_dir.glob("*.txt")):
        all_chunks.extend(chunk_document(txt_file))
    return all_chunks

"""
Chunking for the econ.TH RAG system.

Strategy
--------
Word-window chunking: each paper's text is split on whitespace into words, then
sliced into overlapping windows of ``CHUNK_SIZE`` words (default 400) with
``CHUNK_OVERLAP`` words (default 50) shared between consecutive chunks. We use a
simple word window rather than the model's subword tokenizer because:
  * it is deterministic, fast, and dependency-light;
  * ``all-MiniLM-L6-v2`` accepts up to 256 subword tokens and truncates the rest,
    and ~400 words sits comfortably in that envelope while keeping a small,
    predictable number of chunks per paper. (Words slightly over-shoot the token
    budget, so the model truncates the tail of each chunk; the overlap ensures
    no content is permanently lost across the chunk boundary.)

For each paper we prefer full text (``Paper.full_text()``); if absent we fall
back to title + abstract (``Paper.best_text()``). Each chunk carries metadata
(paper_id, title, authors, primary_category, published, source, chunk_index)
and a deterministic id ``<paper_id>::<chunk_index>::<source>`` so ingestion can
be re-run idempotently.
"""
from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import corpus  # noqa: E402

CHUNK_SIZE = 400  # words per chunk
CHUNK_OVERLAP = 50  # words shared between consecutive chunks


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def split_words(text: str, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word windows. Returns list of chunk strings."""
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_paper(paper: "corpus.Paper", chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Chunk a single paper. Uses full text if available, else title+abstract."""
    full = paper.full_text()
    if full and full.strip():
        text = full
        source = "fulltext"
    else:
        text = paper.best_text()
        source = "abstract"

    pieces = split_words(text, chunk_size, overlap)
    authors = ", ".join(paper.authors) if paper.authors else ""

    chunks: list[Chunk] = []
    for idx, piece in enumerate(pieces):
        meta = {
            "paper_id": paper.id,
            "title": paper.title or "",
            "authors": authors,
            "primary_category": paper.primary_category or "",
            "published": paper.published or "",
            "source": source,
            "chunk_index": idx,
        }
        cid = f"{paper.id}::{idx}::{source}"
        chunks.append(Chunk(id=cid, text=piece, metadata=meta))
    return chunks

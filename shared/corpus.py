"""
Shared corpus access for the econ.TH dataset.

Both the MCP server and the RAG system import this module so they agree on how
the dataset is read. See shared/DATA_CONTRACT.md for the schema.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "data" / "raw" / "metadata.jsonl"
FULLTEXT_DIR = ROOT / "data" / "processed" / "fulltext"


@dataclass
class Paper:
    id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    primary_category: Optional[str]
    published: Optional[str]
    updated: Optional[str]
    abs_url: Optional[str]
    pdf_url: Optional[str]
    doi: Optional[str]
    journal_ref: Optional[str]
    comment: Optional[str]
    has_pdf: bool
    pdf_path: Optional[str]
    fulltext_path: Optional[str]
    raw: dict

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        return cls(
            id=d["id"],
            title=d.get("title", ""),
            authors=d.get("authors", []),
            abstract=d.get("abstract", ""),
            categories=d.get("categories", []),
            primary_category=d.get("primary_category"),
            published=d.get("published"),
            updated=d.get("updated"),
            abs_url=d.get("abs_url"),
            pdf_url=d.get("pdf_url"),
            doi=d.get("doi"),
            journal_ref=d.get("journal_ref"),
            comment=d.get("comment"),
            has_pdf=bool(d.get("has_pdf")),
            pdf_path=d.get("pdf_path"),
            fulltext_path=d.get("fulltext_path"),
            raw=d,
        )

    def full_text(self) -> Optional[str]:
        """Return extracted full text if available, else None."""
        if self.fulltext_path:
            p = ROOT / self.fulltext_path
            if p.exists():
                return p.read_text(errors="ignore")
        p = FULLTEXT_DIR / f"{self.id}.txt"
        if p.exists():
            return p.read_text(errors="ignore")
        return None

    def best_text(self) -> str:
        """Full text if present, otherwise title + abstract."""
        ft = self.full_text()
        if ft:
            return ft
        return f"{self.title}\n\n{self.abstract}"


def load_metadata() -> list[Paper]:
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"{META_PATH} not found. Run scripts/download_arxiv.py first."
        )
    papers: list[Paper] = []
    with META_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(Paper.from_dict(json.loads(line)))
    return papers


def iter_documents() -> Iterator[Paper]:
    yield from load_metadata()


def get_by_id(paper_id: str) -> Optional[Paper]:
    pid = paper_id.split("v")[0]
    for p in load_metadata():
        if p.id == pid:
            return p
    return None


def get_text_for(paper_id: str) -> Optional[str]:
    p = get_by_id(paper_id)
    return p.best_text() if p else None

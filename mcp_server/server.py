#!/usr/bin/env python3
"""
MCP server exposing query/retrieval tools over the arXiv econ.TH corpus.

Transport: stdio. Uses the official `mcp` Python SDK (FastMCP high-level API).

Data is read through the shared `shared/corpus.py` access layer so this server
stays consistent with the RAG component. The corpus is loaded once at startup
and cached; full-text files (which trickle in as the dataset downloads) are read
lazily at query time, so newly extracted text becomes available without restart.

Core logic lives in plain `_impl` functions; the FastMCP `@mcp.tool()` wrappers
delegate to them so the same logic is unit-testable in-process (see
test_server.py) without going over stdio.
"""
from __future__ import annotations

import math
import pathlib
import re
import sys
from collections import Counter
from functools import lru_cache
from typing import Any, Optional

# --- Make the shared data-access layer importable -------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import corpus  # noqa: E402
from shared.corpus import ROOT, Paper  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

MANIFEST_PATH = ROOT / "data" / "processed" / "manifest.json"

mcp = FastMCP("arxiv-econ-th")


# =========================================================================
# Corpus loading / caching
# =========================================================================
@lru_cache(maxsize=1)
def _papers() -> list[Paper]:
    """Load and cache all papers (metadata + abstracts). Metadata is stable."""
    return corpus.load_metadata()


@lru_cache(maxsize=1)
def _index_by_id() -> dict[str, Paper]:
    return {p.id: p for p in _papers()}


# =========================================================================
# Tokenization / scoring helpers
# =========================================================================
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bare_id(paper_id: str) -> str:
    """Strip a trailing version suffix like 'v2' -> bare arXiv id."""
    pid = paper_id.strip()
    # Handle both '2401.01234v2' and 'arxiv:2401.01234v2'
    pid = pid.split(":")[-1]
    return re.sub(r"v\d+$", "", pid)


def _published_date(p: Paper) -> str:
    """Return the YYYY-MM-DD portion of `published`, or '' if missing."""
    if not p.published:
        return ""
    return p.published[:10]


def _make_snippet(text: str, query_terms: list[str], width: int = 280) -> str:
    """Return a snippet centered on the first query-term hit."""
    if not text:
        return ""
    lower = text.lower()
    pos = -1
    for t in query_terms:
        i = lower.find(t)
        if i != -1 and (pos == -1 or i < pos):
            pos = i
    if pos == -1:
        snippet = text[:width]
        suffix = "..." if len(text) > width else ""
        return " ".join(snippet.split()) + suffix
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    snippet = text[start:end]
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + " ".join(snippet.split()) + suffix


# =========================================================================
# Filters
# =========================================================================
def _passes_filters(
    p: Paper,
    category: Optional[str],
    author: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
) -> bool:
    if category:
        cat_l = category.lower()
        if not any(cat_l in c.lower() for c in p.categories):
            return False
    if author:
        auth_l = author.lower()
        if not any(auth_l in a.lower() for a in p.authors):
            return False
    if date_from or date_to:
        d = _published_date(p)
        if not d:
            return False
        if date_from and d < date_from:
            return False
        if date_to and d > date_to:
            return False
    return True


# =========================================================================
# Tool implementations (plain, testable functions)
# =========================================================================
def search_papers_impl(
    query: str,
    max_results: int = 10,
    category: Optional[str] = None,
    author: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict[str, Any]:
    """Rank papers by a BM25-lite score over title + abstract (+ full text)."""
    q_terms = _tokenize(query)
    papers = _papers()

    if not q_terms:
        # No query -> just apply filters and return most-recent first.
        filtered = [
            p
            for p in papers
            if _passes_filters(p, category, author, date_from, date_to)
        ]
        filtered.sort(key=lambda p: p.published or "", reverse=True)
        results = [
            _result_row(p, 0.0, p.abstract, q_terms) for p in filtered[:max_results]
        ]
        return {"query": query, "count": len(results), "results": results}

    q_set = set(q_terms)
    N = len(papers)

    # Document frequency over title+abstract tokens (cheap, metadata-only).
    candidates: list[tuple[Paper, list[str], str]] = []
    df: Counter[str] = Counter()
    for p in papers:
        if not _passes_filters(p, category, author, date_from, date_to):
            continue
        searchable = f"{p.title}\n{p.abstract}"
        # Cheaply fold in full text when present (lazy read).
        ft = p.full_text()
        if ft:
            searchable = f"{searchable}\n{ft}"
        toks = _tokenize(searchable)
        candidates.append((p, toks, searchable))
        for t in q_set:
            if t in toks:
                df[t] += 1

    if not candidates:
        return {"query": query, "count": 0, "results": []}

    # BM25-lite parameters.
    k1, b = 1.5, 0.75
    avgdl = sum(len(toks) for _, toks, _ in candidates) / len(candidates)
    title_boost = 2.5

    scored: list[tuple[float, Paper, str]] = []
    for p, toks, searchable in candidates:
        if not toks:
            continue
        tf = Counter(toks)
        title_toks = set(_tokenize(p.title))
        dl = len(toks)
        score = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if f == 0:
                continue
            n_t = df.get(t, 0)
            idf = math.log(1 + (N - n_t + 0.5) / (n_t + 0.5))
            denom = f + k1 * (1 - b + b * dl / avgdl)
            term_score = idf * (f * (k1 + 1)) / denom
            if t in title_toks:
                term_score *= title_boost
            score += term_score
        if score > 0:
            scored.append((score, p, searchable))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_results]
    results = [
        _result_row(p, round(score, 4), searchable, q_terms)
        for score, p, searchable in top
    ]
    return {"query": query, "count": len(results), "results": results}


def _result_row(p: Paper, score: float, searchable: str, q_terms: list[str]) -> dict:
    return {
        "id": p.id,
        "title": p.title,
        "authors": p.authors,
        "primary_category": p.primary_category,
        "published": p.published,
        "score": score,
        "snippet": _make_snippet(searchable, q_terms),
        "abs_url": p.abs_url,
        "has_full_text": p.full_text() is not None,
    }


def get_paper_impl(paper_id: str) -> dict[str, Any]:
    pid = _bare_id(paper_id)
    p = _index_by_id().get(pid)
    if p is None:
        return {"error": f"No paper found with id '{paper_id}' (bare id '{pid}')."}
    return {
        "id": p.id,
        "version_id": p.raw.get("version_id"),
        "title": p.title,
        "authors": p.authors,
        "abstract": p.abstract,
        "categories": p.categories,
        "primary_category": p.primary_category,
        "published": p.published,
        "updated": p.updated,
        "abs_url": p.abs_url,
        "pdf_url": p.pdf_url,
        "doi": p.doi,
        "journal_ref": p.journal_ref,
        "comment": p.comment,
        "has_full_text": p.full_text() is not None,
    }


def get_full_text_impl(paper_id: str, max_chars: int = 20000) -> dict[str, Any]:
    pid = _bare_id(paper_id)
    p = _index_by_id().get(pid)
    if p is None:
        return {"error": f"No paper found with id '{paper_id}' (bare id '{pid}')."}
    ft = p.full_text()
    if ft is None:
        return {
            "id": p.id,
            "title": p.title,
            "full_text_available": False,
            "message": (
                "Full text is not available for this paper (no PDF was "
                "downloaded). Returning the abstract instead."
            ),
            "abstract": p.abstract,
        }
    total = len(ft)
    truncated = total > max_chars
    body = ft[:max_chars]
    note = (
        f"[Truncated to {max_chars} of {total} characters.]" if truncated else None
    )
    return {
        "id": p.id,
        "title": p.title,
        "full_text_available": True,
        "truncated": truncated,
        "total_chars": total,
        "returned_chars": len(body),
        "note": note,
        "text": body,
    }


def list_categories_impl() -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for p in _papers():
        for c in p.categories:
            counts[c] += 1
    categories = [
        {"category": c, "count": n} for c, n in counts.most_common()
    ]
    return {"total_distinct": len(categories), "categories": categories}


def get_corpus_stats_impl() -> dict[str, Any]:
    papers = _papers()
    n = len(papers)
    with_text = sum(1 for p in papers if p.full_text() is not None)

    dates = [_published_date(p) for p in papers if _published_date(p)]
    date_range = {
        "earliest": min(dates) if dates else None,
        "latest": max(dates) if dates else None,
    }

    author_counts: Counter[str] = Counter()
    cat_counts: Counter[str] = Counter()
    for p in papers:
        for a in p.authors:
            author_counts[a] += 1
        for c in p.categories:
            cat_counts[c] += 1

    stats: dict[str, Any] = {
        "total_papers": n,
        "papers_with_full_text": with_text,
        "papers_with_pdf_flag": sum(1 for p in papers if p.has_pdf),
        "date_range": date_range,
        "top_authors": [
            {"author": a, "count": c} for a, c in author_counts.most_common(10)
        ],
        "top_categories": [
            {"category": c, "count": n_} for c, n_ in cat_counts.most_common(10)
        ],
    }

    # Fold in the on-disk manifest summary if present.
    if MANIFEST_PATH.exists():
        try:
            import json

            stats["manifest"] = json.loads(MANIFEST_PATH.read_text())
        except Exception as e:  # pragma: no cover - defensive
            stats["manifest_error"] = str(e)
    return stats


def get_recent_papers_impl(n: int = 10, category: Optional[str] = None) -> dict[str, Any]:
    papers = _papers()
    if category:
        cat_l = category.lower()
        papers = [
            p for p in papers if any(cat_l in c.lower() for c in p.categories)
        ]
    papers = sorted(papers, key=lambda p: p.published or "", reverse=True)
    rows = [
        {
            "id": p.id,
            "title": p.title,
            "authors": p.authors,
            "primary_category": p.primary_category,
            "published": p.published,
            "abs_url": p.abs_url,
        }
        for p in papers[:n]
    ]
    return {"count": len(rows), "results": rows}


# =========================================================================
# FastMCP tool wrappers (these are what the LLM client sees)
# =========================================================================
@mcp.tool()
def search_papers(
    query: str,
    max_results: int = 10,
    category: Optional[str] = None,
    author: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    """Keyword/relevance search over the econ.TH corpus.

    Ranks papers by a BM25-lite relevance score computed over each paper's
    title and abstract (and full text when it has been extracted), with a
    boost for query terms appearing in the title. Use this to find papers on a
    topic, e.g. "auction design", "mechanism without transfers", "equilibrium
    refinement".

    Args:
        query: Free-text search terms (matched against title + abstract + full text).
        max_results: Maximum number of ranked results to return (default 10).
        category: Optional filter; substring-matched against each paper's
            arXiv categories (e.g. "econ.TH", "math.OC", "cs.GT").
        author: Optional filter; substring-matched against author names
            (case-insensitive).
        date_from: Optional inclusive lower bound on publication date,
            ISO format "YYYY-MM-DD".
        date_to: Optional inclusive upper bound on publication date,
            ISO format "YYYY-MM-DD".

    Returns:
        {query, count, results:[{id, title, authors, primary_category,
        published, score, snippet, abs_url, has_full_text}]} ranked by score.
    """
    return search_papers_impl(
        query, max_results, category, author, date_from, date_to
    )


@mcp.tool()
def get_paper(paper_id: str) -> dict:
    """Return full metadata for a single paper by its arXiv id.

    Accepts a bare id ("2401.01234") or a versioned id ("2401.01234v2"); the
    version suffix is stripped automatically. Returns title, authors, abstract,
    categories, dates, links (abs/pdf), DOI, journal reference, and comment.

    Args:
        paper_id: arXiv identifier, with or without a version suffix.

    Returns:
        A metadata dict, or {error: ...} if no matching paper exists.
    """
    return get_paper_impl(paper_id)


@mcp.tool()
def get_full_text(paper_id: str, max_chars: int = 20000) -> dict:
    """Return the extracted full text of a paper, if available.

    Many papers in this corpus have only an abstract (their PDF was not
    downloaded). When full text exists it is returned truncated to `max_chars`
    with a note about truncation. When it does not exist, a clear message is
    returned along with the abstract as a fallback.

    Args:
        paper_id: arXiv identifier, with or without a version suffix.
        max_chars: Maximum characters of full text to return (default 20000).

    Returns:
        If text exists: {id, title, full_text_available: true, truncated,
        total_chars, returned_chars, note, text}. Otherwise:
        {id, title, full_text_available: false, message, abstract}.
    """
    return get_full_text_impl(paper_id, max_chars)


@mcp.tool()
def list_categories() -> dict:
    """List the distinct arXiv categories in the corpus with paper counts.

    Counts every category each paper is tagged with (a paper may appear under
    several), sorted by descending count. Useful for discovering which
    sub-areas (e.g. cs.GT, math.OC, q-fin.EC) co-occur with econ.TH.

    Returns:
        {total_distinct, categories:[{category, count}]} sorted desc by count.
    """
    return list_categories_impl()


@mcp.tool()
def get_corpus_stats() -> dict:
    """Return summary statistics for the whole corpus.

    Includes total number of papers, how many currently have extracted full
    text (this grows as the dataset finishes downloading), the publication date
    range, and the top authors and categories. Also includes the on-disk
    manifest summary when present.

    Returns:
        {total_papers, papers_with_full_text, papers_with_pdf_flag,
        date_range:{earliest, latest}, top_authors, top_categories, manifest}.
    """
    return get_corpus_stats_impl()


@mcp.tool()
def get_recent_papers(n: int = 10, category: Optional[str] = None) -> dict:
    """Return the most recently published papers.

    Sorted by publication date, newest first.

    Args:
        n: Number of papers to return (default 10).
        category: Optional filter; substring-matched against arXiv categories.

    Returns:
        {count, results:[{id, title, authors, primary_category, published,
        abs_url}]}.
    """
    return get_recent_papers_impl(n, category)


if __name__ == "__main__":
    mcp.run()

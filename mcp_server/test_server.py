#!/usr/bin/env python3
"""
Self-contained smoke test for the arxiv-econ-th MCP server.

Calls each tool's underlying plain `_impl` function directly with real data
(NOT over stdio) and prints results. Exits non-zero on any failure.

Run:
    cd "<project root>" && ./.venv/bin/python mcp_server/test_server.py
"""
from __future__ import annotations

import sys
import traceback

import server


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {msg}")
        if not cond:
            failures.append(msg)

    # --- corpus stats -----------------------------------------------------
    _section("get_corpus_stats")
    try:
        stats = server.get_corpus_stats_impl()
        print(f"  total_papers          = {stats['total_papers']}")
        print(f"  papers_with_full_text = {stats['papers_with_full_text']}")
        print(f"  date_range            = {stats['date_range']}")
        print(f"  top_categories[:3]    = {stats['top_categories'][:3]}")
        check(stats["total_papers"] > 1000, "corpus has > 1000 papers")
        check("manifest" in stats, "manifest summary included")
    except Exception:
        traceback.print_exc()
        failures.append("get_corpus_stats raised")

    # --- list categories --------------------------------------------------
    _section("list_categories")
    try:
        cats = server.list_categories_impl()
        print(f"  total_distinct = {cats['total_distinct']}")
        print(f"  top 5          = {cats['categories'][:5]}")
        check(cats["total_distinct"] > 0, "has >=1 distinct category")
        check(
            any(c["category"] == "econ.TH" for c in cats["categories"]),
            "econ.TH present in categories",
        )
    except Exception:
        traceback.print_exc()
        failures.append("list_categories raised")

    # --- search: equilibrium ----------------------------------------------
    _section('search_papers("equilibrium")')
    eq_id = None
    try:
        res = server.search_papers_impl("equilibrium", max_results=5)
        print(f"  count = {res['count']}")
        for r in res["results"]:
            print(f"   [{r['score']:>7}] {r['id']}  {r['title'][:60]}")
        check(res["count"] > 0, "equilibrium search returns results")
        if res["results"]:
            eq_id = res["results"][0]["id"]
            check(
                res["results"][0]["score"] > 0, "top result has positive score"
            )
            check(
                bool(res["results"][0]["snippet"]), "top result has a snippet"
            )
    except Exception:
        traceback.print_exc()
        failures.append("search_papers(equilibrium) raised")

    # --- search: auction --------------------------------------------------
    _section('search_papers("auction")')
    try:
        res = server.search_papers_impl("auction", max_results=5)
        print(f"  count = {res['count']}")
        for r in res["results"]:
            print(f"   [{r['score']:>7}] {r['id']}  {r['title'][:60]}")
        check(res["count"] > 0, "auction search returns results")
    except Exception:
        traceback.print_exc()
        failures.append("search_papers(auction) raised")

    # --- search with filters ----------------------------------------------
    _section('search_papers("equilibrium", category="econ.TH", date_from)')
    try:
        res = server.search_papers_impl(
            "equilibrium", max_results=5, category="econ.TH", date_from="2020-01-01"
        )
        print(f"  count = {res['count']}")
        ok = all(
            (r["published"] or "")[:10] >= "2020-01-01" for r in res["results"]
        )
        check(ok, "all filtered results respect date_from")
    except Exception:
        traceback.print_exc()
        failures.append("search_papers(filtered) raised")

    # --- get_paper --------------------------------------------------------
    _section("get_paper")
    try:
        if eq_id is None:
            raise RuntimeError("no id from earlier search")
        p = server.get_paper_impl(eq_id)
        print(f"  id     = {p['id']}")
        print(f"  title  = {p['title'][:70]}")
        print(f"  authors= {p['authors'][:3]}")
        check(p.get("id") == eq_id, "get_paper returns requested paper")
        check(bool(p.get("abstract")), "paper has an abstract")
        # version-suffix handling
        pv = server.get_paper_impl(eq_id + "v1")
        check(pv.get("id") == eq_id, "version-suffixed id resolves to bare id")
        # missing id
        miss = server.get_paper_impl("0000.00000")
        check("error" in miss, "missing id returns an error field")
    except Exception:
        traceback.print_exc()
        failures.append("get_paper raised")

    # --- get_full_text: find one that has text ----------------------------
    _section("get_full_text")
    try:
        papers = server._papers()
        with_text = next((p for p in papers if p.full_text() is not None), None)
        if with_text is not None:
            ft = server.get_full_text_impl(with_text.id, max_chars=1000)
            print(f"  id                  = {ft['id']}")
            print(f"  full_text_available = {ft['full_text_available']}")
            print(f"  returned_chars      = {ft.get('returned_chars')}")
            print(f"  truncated           = {ft.get('truncated')}")
            print(f"  text[:120]          = {ft.get('text','')[:120]!r}")
            check(ft["full_text_available"] is True, "full text returned for PDF paper")
            check(len(ft.get("text", "")) <= 1000, "respects max_chars truncation")
            check(bool(ft.get("text")), "returned non-empty text")
        else:
            print("  (no paper with extracted full text yet — skipping positive case)")

        # negative case: a paper without full text
        no_text = next((p for p in papers if p.full_text() is None), None)
        if no_text is not None:
            ft0 = server.get_full_text_impl(no_text.id)
            print(f"  no-text paper {no_text.id}: available={ft0['full_text_available']}")
            check(
                ft0["full_text_available"] is False,
                "no-text paper reports unavailable",
            )
            check(bool(ft0.get("abstract")), "no-text paper falls back to abstract")
    except Exception:
        traceback.print_exc()
        failures.append("get_full_text raised")

    # --- get_recent_papers ------------------------------------------------
    _section("get_recent_papers")
    try:
        rec = server.get_recent_papers_impl(n=5)
        print(f"  count = {rec['count']}")
        for r in rec["results"]:
            print(f"   {r['published']}  {r['id']}  {r['title'][:55]}")
        check(rec["count"] > 0, "recent papers returned")
        pubs = [r["published"] or "" for r in rec["results"]]
        check(pubs == sorted(pubs, reverse=True), "recent papers sorted desc by date")
    except Exception:
        traceback.print_exc()
        failures.append("get_recent_papers raised")

    # --- verify FastMCP registration --------------------------------------
    _section("FastMCP tool registration")
    try:
        import asyncio

        tools = asyncio.run(server.mcp.list_tools())
        names = {t.name for t in tools}
        print(f"  registered tools = {sorted(names)}")
        expected = {
            "search_papers",
            "get_paper",
            "get_full_text",
            "list_categories",
            "get_corpus_stats",
            "get_recent_papers",
        }
        check(expected.issubset(names), "all 6 tools registered with FastMCP")
    except Exception:
        traceback.print_exc()
        failures.append("FastMCP registration check raised")

    # --- summary ----------------------------------------------------------
    _section("SUMMARY")
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

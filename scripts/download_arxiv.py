#!/usr/bin/env python3
"""
Download arXiv econ.TH (Theoretical Economics) dataset.

Stage 1: fetch metadata + abstracts for ALL econ.TH papers via the arXiv API
         -> data/raw/metadata.jsonl
Stage 2: download full-text PDFs, most-recent first, until a cumulative size
         cap (default ~0.95 GB) -> data/raw/pdfs/<id>.pdf

The script is resumable: existing PDFs are skipped, and metadata can be skipped
with --skip-metadata. It is polite to arXiv (descriptive User-Agent + delay).

Usage:
    python scripts/download_arxiv.py                 # full run
    python scripts/download_arxiv.py --metadata-only
    python scripts/download_arxiv.py --max-gb 0.95
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

API_URL = "https://export.arxiv.org/api/query"
CATEGORY = "econ.TH"
USER_AGENT = (
    "arxiv-econ-th-research/1.0 (+local RAG/MCP research project; "
    "mailto:jtalface@gmail.com)"
)

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PDF_DIR = RAW / "pdfs"
META_PATH = RAW / "metadata.jsonl"
MANIFEST_PATH = DATA / "processed" / "manifest.json"


def _text(node, path):
    el = node.find(path, NS)
    return el.text.strip() if el is not None and el.text else None


def parse_entry(entry) -> dict | None:
    raw_id = _text(entry, "atom:id")  # http://arxiv.org/abs/2401.01234v2
    if not raw_id:
        return None
    abs_url = raw_id
    arxiv_id = raw_id.rsplit("/abs/", 1)[-1]  # 2401.01234v2
    base_id = arxiv_id.split("v")[0]          # 2401.01234

    authors = [
        a.text.strip()
        for a in entry.findall("atom:author/atom:name", NS)
        if a.text
    ]
    categories = [
        c.get("term") for c in entry.findall("atom:category", NS) if c.get("term")
    ]
    primary = entry.find("arxiv:primary_category", NS)
    primary_cat = primary.get("term") if primary is not None else None

    pdf_url = None
    for link in entry.findall("atom:link", NS):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = link.get("href")
    if pdf_url is None:
        pdf_url = f"https://arxiv.org/pdf/{base_id}"

    return {
        "id": base_id,
        "version_id": arxiv_id,
        "title": " ".join((_text(entry, "atom:title") or "").split()),
        "authors": authors,
        "abstract": " ".join((_text(entry, "atom:summary") or "").split()),
        "categories": categories,
        "primary_category": primary_cat,
        "published": _text(entry, "atom:published"),
        "updated": _text(entry, "atom:updated"),
        "abs_url": abs_url,
        "pdf_url": pdf_url,
        "doi": _text(entry, "arxiv:doi"),
        "journal_ref": _text(entry, "arxiv:journal_ref"),
        "comment": _text(entry, "arxiv:comment"),
        # filled in stage 2 / extraction:
        "has_pdf": False,
        "pdf_path": None,
        "fulltext_path": None,
    }


def fetch_metadata(session, page_size=100, delay=3.0) -> list[dict]:
    print(f"[meta] fetching econ.TH metadata (page_size={page_size})...")
    records: list[dict] = []
    start = 0
    total = None
    while True:
        params = {
            "search_query": f"cat:{CATEGORY}",
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        r = None
        for attempt in range(8):
            try:
                r = session.get(API_URL, params=params, timeout=60)
                if r.status_code == 429:
                    raise requests.HTTPError("429 rate limited")
                r.raise_for_status()
                break
            except Exception as e:  # noqa: BLE001
                r = None
                wait = min(15 * (attempt + 1), 90)
                print(f"[meta] error {e}; retry in {wait:.0f}s")
                time.sleep(wait)
        if r is None:
            print(f"[meta] giving up at start={start} after retries")
            break

        feed = ET.fromstring(r.content)
        if total is None:
            tr = feed.find("opensearch:totalResults", NS)
            total = int(tr.text) if tr is not None else 0
            print(f"[meta] totalResults={total}")

        entries = feed.findall("atom:entry", NS)
        if not entries:
            break
        for e in entries:
            rec = parse_entry(e)
            if rec:
                records.append(rec)
        print(f"[meta] {len(records)}/{total}")
        start += page_size
        if total and start >= total:
            break
        time.sleep(delay)
    return records


def download_pdfs(session, records, max_bytes, delay=3.0):
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    # most recent first (records already sorted descending by submittedDate)
    cumulative = sum(p.stat().st_size for p in PDF_DIR.glob("*.pdf"))
    print(f"[pdf] existing on disk: {cumulative/1e6:.1f} MB")
    by_id = {r["id"]: r for r in records}

    downloaded = 0
    for rec in records:
        if cumulative >= max_bytes:
            print(f"[pdf] reached cap {max_bytes/1e9:.2f} GB; stopping")
            break
        out = PDF_DIR / f"{rec['id']}.pdf"
        if out.exists() and out.stat().st_size > 0:
            rec["has_pdf"] = True
            rec["pdf_path"] = str(out.relative_to(ROOT))
            continue
        url = rec["pdf_url"]
        try:
            r = session.get(url, timeout=120, stream=True)
            r.raise_for_status()
            content = r.content
            if not content.startswith(b"%PDF"):
                print(f"[pdf] {rec['id']}: not a PDF, skipping")
                time.sleep(delay)
                continue
            out.write_bytes(content)
            size = len(content)
            cumulative += size
            downloaded += 1
            rec["has_pdf"] = True
            rec["pdf_path"] = str(out.relative_to(ROOT))
            print(
                f"[pdf] {downloaded:>4} {rec['id']} {size/1e6:5.2f}MB "
                f"total={cumulative/1e9:.3f}GB"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[pdf] {rec['id']}: error {e}")
        time.sleep(delay)

    # keep by_id reference for callers
    return by_id, cumulative


def write_metadata(records):
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[meta] wrote {len(records)} records -> {META_PATH}")


def load_metadata() -> list[dict]:
    records = []
    with META_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_manifest(records, cumulative):
    n_pdf = sum(1 for r in records if r.get("has_pdf"))
    manifest = {
        "category": CATEGORY,
        "total_papers": len(records),
        "papers_with_pdf": n_pdf,
        "pdf_bytes": cumulative,
        "pdf_gb": round(cumulative / 1e9, 3),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] {json.dumps(manifest)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=0.95,
                    help="cumulative PDF size cap in GB")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="seconds between requests (arXiv etiquette)")
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--skip-metadata", action="store_true",
                    help="reuse existing metadata.jsonl, only download PDFs")
    ap.add_argument("--no-pdfs", action="store_true")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    if args.skip_metadata and META_PATH.exists():
        records = load_metadata()
        print(f"[meta] loaded {len(records)} existing records")
    else:
        records = fetch_metadata(session, delay=args.delay)
        write_metadata(records)

    cumulative = 0
    if not args.metadata_only and not args.no_pdfs:
        _, cumulative = download_pdfs(
            session, records, int(args.max_gb * 1e9), delay=args.delay
        )
        # persist has_pdf / pdf_path flags back into metadata
        write_metadata(records)

    write_manifest(records, cumulative)
    print("[done]")


if __name__ == "__main__":
    main()

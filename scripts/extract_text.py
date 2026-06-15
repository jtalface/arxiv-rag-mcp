#!/usr/bin/env python3
"""
Extract plain text from downloaded econ.TH PDFs.

Reads data/raw/pdfs/<id>.pdf -> writes data/processed/fulltext/<id>.txt and
updates the `fulltext_path` field in data/raw/metadata.jsonl. Resumable: skips
PDFs whose .txt already exists.

Usage:
    python scripts/extract_text.py
"""
from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "raw" / "pdfs"
OUT_DIR = ROOT / "data" / "processed" / "fulltext"
META_PATH = ROOT / "data" / "raw" / "metadata.jsonl"


def extract_one(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            continue
    text = "\n".join(parts)
    # light normalization: collapse runs of blank lines
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned, blanks = [], 0
    for ln in lines:
        if ln.strip():
            blanks = 0
            cleaned.append(ln)
        else:
            blanks += 1
            if blanks <= 1:
                cleaned.append("")
    return "\n".join(cleaned).strip()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"[extract] {len(pdfs)} PDFs found")

    done = 0
    for pdf in pdfs:
        out = OUT_DIR / f"{pdf.stem}.txt"
        if out.exists() and out.stat().st_size > 0:
            continue
        try:
            text = extract_one(pdf)
            if len(text) < 200:  # likely scanned / failed extraction
                print(f"[extract] {pdf.stem}: short text ({len(text)} chars)")
            out.write_text(text)
            done += 1
            if done % 50 == 0:
                print(f"[extract] {done} extracted...")
        except Exception as e:  # noqa: BLE001
            print(f"[extract] {pdf.stem}: error {e}")

    # update metadata fulltext_path
    if META_PATH.exists():
        records = []
        with META_PATH.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        for r in records:
            txt = OUT_DIR / f"{r['id']}.txt"
            if txt.exists() and txt.stat().st_size > 0:
                r["fulltext_path"] = str(txt.relative_to(ROOT))
        with META_PATH.open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n = sum(1 for r in records if r.get("fulltext_path"))
        print(f"[extract] metadata updated; {n} records have fulltext")

    print(f"[extract] done; {done} new extractions")


if __name__ == "__main__":
    main()

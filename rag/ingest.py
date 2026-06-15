"""
Ingest the econ.TH corpus into a persistent Chroma vector store.

Pipeline: load corpus (shared.corpus) -> chunk each paper (chunking.py) ->
embed locally (embeddings.py) -> upsert into a Chroma collection at
``rag/chroma_db/``.

Idempotent: chunk ids are deterministic (``<paper_id>::<chunk_index>::<source>``)
and we use ``collection.upsert``, so re-running adds only new/changed chunks and
never duplicates. Re-run later to pick up full-text files that finish downloading.

Usage:
    python rag/ingest.py                # ingest all papers
    python rag/ingest.py --limit 200    # ingest first 200 papers (quick test)
    python rag/ingest.py --reset        # drop & rebuild the collection
"""
from __future__ import annotations

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import corpus  # noqa: E402

from chunking import chunk_paper  # noqa: E402
from embeddings import embed_texts, MODEL_NAME, EMBED_DIM  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DB_PATH = HERE / "chroma_db"
COLLECTION_NAME = "econ_th"
EMBED_BATCH = 256


def get_collection(reset: bool = False):
    import chromadb
    client = chromadb.PersistentClient(path=str(DB_PATH))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[reset] dropped existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": MODEL_NAME,
            "embedding_dim": EMBED_DIM,
        },
    )


def ingest(limit: int | None = None, reset: bool = False) -> None:
    collection = get_collection(reset=reset)
    start_count = collection.count()
    print(f"Collection '{COLLECTION_NAME}' at {DB_PATH} (start count: {start_count})")

    papers = corpus.load_metadata()
    if limit is not None:
        papers = papers[:limit]
    print(f"Loaded {len(papers)} papers from corpus.")

    pending_ids: list[str] = []
    pending_docs: list[str] = []
    pending_meta: list[dict] = []
    total_chunks = 0
    fulltext_papers = 0

    def flush() -> None:
        nonlocal pending_ids, pending_docs, pending_meta
        if not pending_ids:
            return
        embeddings = embed_texts(pending_docs, batch_size=EMBED_BATCH)
        collection.upsert(
            ids=pending_ids,
            documents=pending_docs,
            embeddings=embeddings,
            metadatas=pending_meta,
        )
        pending_ids, pending_docs, pending_meta = [], [], []

    for i, paper in enumerate(papers, 1):
        chunks = chunk_paper(paper)
        if not chunks:
            continue
        if chunks[0].metadata["source"] == "fulltext":
            fulltext_papers += 1
        for ch in chunks:
            pending_ids.append(ch.id)
            pending_docs.append(ch.text)
            pending_meta.append(ch.metadata)
            total_chunks += 1
        if len(pending_ids) >= EMBED_BATCH:
            flush()
        if i % 100 == 0 or i == len(papers):
            print(f"  processed {i}/{len(papers)} papers, {total_chunks} chunks queued/written")

    flush()

    end_count = collection.count()
    print("-" * 60)
    print(f"Done. Papers processed: {len(papers)} "
          f"({fulltext_papers} with full text, {len(papers) - fulltext_papers} abstract-only)")
    print(f"Chunks generated this run: {total_chunks}")
    print(f"Collection count: {start_count} -> {end_count} "
          f"(net new: {end_count - start_count})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest econ.TH corpus into Chroma.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only ingest the first N papers (quick test).")
    ap.add_argument("--reset", action="store_true",
                    help="Drop and rebuild the collection before ingesting.")
    args = ap.parse_args()
    ingest(limit=args.limit, reset=args.reset)


if __name__ == "__main__":
    main()

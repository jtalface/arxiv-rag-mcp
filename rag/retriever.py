"""
Retrieval over the persistent Chroma econ.TH collection.

Embeds the query with the same local ``all-MiniLM-L6-v2`` model used at ingest
time and queries Chroma, with optional metadata filters on ``source``
("fulltext" / "abstract") and ``primary_category``.
"""
from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from embeddings import embed_query  # noqa: E402

DB_PATH = HERE / "chroma_db"
COLLECTION_NAME = "econ_th"


@dataclass
class Hit:
    chunk_id: str
    text: str
    distance: float
    score: float  # cosine similarity (1 - distance)
    metadata: dict

    @property
    def paper_id(self) -> str:
        return self.metadata.get("paper_id", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")


_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(DB_PATH))
        _collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def retrieve(query: str, k: int = 5, source: str | None = None,
             category: str | None = None) -> list[Hit]:
    """Retrieve top-k chunks for a query, with optional metadata filters."""
    collection = _get_collection()

    conditions = []
    if source:
        conditions.append({"source": source})
    if category:
        conditions.append({"primary_category": category})
    where: dict | None = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    q_emb = embed_query(query)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[Hit] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        hits.append(Hit(
            chunk_id=cid,
            text=doc,
            distance=dist,
            score=1.0 - dist,
            metadata=meta or {},
        ))
    return hits

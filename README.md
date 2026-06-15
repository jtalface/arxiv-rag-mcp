# arXiv econ.TH — MCP + RAG over Theoretical Economics papers

Two systems built on one shared dataset downloaded from arXiv's
**Economics → Theoretical Economics (`econ.TH`)** category:

1. **MCP server** (`mcp_server/`) — exposes tools to query and retrieve papers
   (search by keyword/author/category/date, fetch metadata, fetch full text).
2. **RAG system** (`rag/`) — chunks + embeds the corpus into a Chroma vector DB
   (local `sentence-transformers` embeddings) and answers semantic queries with
   retrieved context.

Both read the same source data via `shared/corpus.py`. See
[`shared/DATA_CONTRACT.md`](shared/DATA_CONTRACT.md) for the schema.

## Layout

```
data/raw/metadata.jsonl        # all econ.TH papers (metadata + abstracts)
data/raw/pdfs/<id>.pdf         # full-text PDFs (size-capped ~500 MB)
data/processed/fulltext/<id>.txt
data/processed/manifest.json
scripts/download_arxiv.py      # stage 1: metadata, stage 2: PDFs (capped)
scripts/extract_text.py        # PDF -> text
shared/corpus.py               # shared dataset access layer
mcp_server/                    # MCP server (see its README)
rag/                           # RAG pipeline + query (see its README)
```

## Setup

```bash
python3.13 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Build the dataset

```bash
./.venv/bin/python scripts/download_arxiv.py --max-gb 0.5    # ~500 MB cap
./.venv/bin/python scripts/extract_text.py
```

## Run

- MCP server: see [`mcp_server/README.md`](mcp_server/README.md)
- RAG: see [`rag/README.md`](rag/README.md)

# Shared Data Contract — arXiv econ.TH corpus

Both the **MCP server** (`mcp_server/`) and the **RAG system** (`rag/`) read from the
same source data produced by `scripts/download_arxiv.py` and
`scripts/extract_text.py`. Treat the paths and schema below as a fixed interface.

## Paths (relative to project root)

| Path | What |
|------|------|
| `data/raw/metadata.jsonl` | One JSON object per paper, ALL econ.TH papers (~4046). |
| `data/raw/pdfs/<id>.pdf` | Downloaded PDFs (subset, size-capped ~500 MB). |
| `data/processed/fulltext/<id>.txt` | Extracted plain text for papers that have a PDF. |
| `data/processed/manifest.json` | Corpus summary stats. |

`<id>` is the bare arXiv id, e.g. `2401.01234` (no version suffix, no slashes).

## `metadata.jsonl` record schema

```json
{
  "id": "2401.01234",
  "version_id": "2401.01234v2",
  "title": "string",
  "authors": ["First Last", "..."],
  "abstract": "string (whitespace-normalized)",
  "categories": ["econ.TH", "math.OC"],
  "primary_category": "econ.TH",
  "published": "2024-01-02T18:00:00Z",
  "updated": "2024-03-01T12:00:00Z",
  "abs_url": "http://arxiv.org/abs/2401.01234v2",
  "pdf_url": "https://arxiv.org/pdf/2401.01234",
  "doi": "10.xxxx/yyy or null",
  "journal_ref": "string or null",
  "comment": "string or null",
  "has_pdf": true,
  "pdf_path": "data/raw/pdfs/2401.01234.pdf or null",
  "fulltext_path": "data/processed/fulltext/2401.01234.txt or null"
}
```

Notes:
- `has_pdf` / `pdf_path` / `fulltext_path` may be `false`/`null` for papers whose
  PDF was not downloaded (beyond the size cap). EVERY paper has full metadata +
  abstract regardless.
- `abstract` is always present and is the primary text field for papers without
  full text.

## Contract rules for both components

1. **Read-only** on `data/`. Do not mutate metadata or PDFs.
2. Each component owns its own subdir (`mcp_server/` or `rag/`) and its own
   persistence (e.g. `rag/chroma_db/`). Do not write into the other's dir.
3. Use the project venv at `.venv/` (Python 3.13). Add deps to your component's
   own `requirements.txt`; do not remove others' deps.
4. Degrade gracefully when `has_pdf` is false — fall back to title + abstract.
5. Load the corpus via the shared helper `shared/corpus.py` (load_metadata,
   iter_documents, get_text_for) so both components stay consistent.

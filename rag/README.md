# econ.TH RAG System

A local, offline Retrieval-Augmented Generation (RAG) system over the arXiv
**Theoretical Economics (`econ.TH`)** corpus. It chunks each paper, embeds the
chunks with a local model, stores the vectors in a persistent Chroma database,
and assembles cited, augmented prompts for any LLM.

## Architecture

```
corpus (shared/corpus.py)        ~4046 papers: metadata + abstracts,
        │                        plus full text for papers with a PDF
        ▼
chunk      (chunking.py)         word-window chunks, ~400 words / 50 overlap
        │                        prefer full text, fall back to title+abstract
        ▼
embed      (embeddings.py)       LOCAL sentence-transformers all-MiniLM-L6-v2
        │                        384-dim, normalized — no API, no network
        ▼
store      (ingest.py)           Chroma PersistentClient, collection "econ_th",
        │                        cosine space, deterministic chunk ids (upsert)
        ▼
retrieve   (retriever.py)        embed query → Chroma top-k + metadata filters
        │
        ▼
augment    (query.py)            CONTEXT (cited) + QUESTION + ANSWER PROMPT
                                 → ready to feed to an LLM
```

### Components
- `chunking.py` — `chunk_paper(paper)` splits text into overlapping word windows
  (`CHUNK_SIZE=400`, `CHUNK_OVERLAP=50`). Each chunk gets metadata
  (`paper_id`, `title`, `authors`, `primary_category`, `published`, `source`,
  `chunk_index`) and a deterministic id `<paper_id>::<chunk_index>::<source>`.
- `embeddings.py` — singleton wrapper around `all-MiniLM-L6-v2` (384-dim);
  `embed_texts(list[str]) -> list[vector]`. Fully local/offline.
- `ingest.py` — CLI that loads the corpus, chunks, embeds, and **upserts** into
  Chroma. Idempotent and incremental.
- `retriever.py` — `retrieve(query, k=5, source=None, category=None) -> list[Hit]`.
- `query.py` — CLI + `augment(query, k)` that builds the augmented RAG prompt.

## Chunking strategy
Simple whitespace word windows: split text into words, slice into 400-word
windows with 50-word overlap. Deterministic and dependency-light. `all-MiniLM-L6-v2`
caps input at 256 subword tokens and truncates beyond that; the 50-word overlap
ensures content near a chunk boundary still appears whole in a neighbor chunk.
Papers with extracted full text are chunked from full text (`source="fulltext"`);
papers without are chunked from title + abstract (`source="abstract"`).

## Embeddings — local & offline
Model: `sentence-transformers` **`all-MiniLM-L6-v2`**, **384** dimensions,
normalized. Loaded once as a process singleton. No API keys and no external
embedding calls — everything runs on the local machine from the cached model.

## Vector store
Chroma `PersistentClient`, collection **`econ_th`**, cosine similarity, stored at:

```
rag/chroma_db/
```

## Usage

All commands run from the project root with the project venv.

### Ingest
```bash
# quick test (first 200 papers)
./.venv/bin/python rag/ingest.py --limit 200

# full corpus
./.venv/bin/python rag/ingest.py

# drop & rebuild the collection
./.venv/bin/python rag/ingest.py --reset
```

Ingestion is **idempotent and incremental**: chunk ids are deterministic and
written with `collection.upsert`, so re-running adds only new/changed chunks and
never duplicates. Because the full-text `.txt` files download over time, re-run
`ingest.py` later to pick up newly available full text — a paper that was
`abstract`-only is re-chunked as `fulltext` once its text arrives (new ids under
the `::fulltext` source).

### Query
```bash
./.venv/bin/python rag/query.py "mechanism design with incomplete information"
./.venv/bin/python rag/query.py "auction revenue" --k 8 --source fulltext
./.venv/bin/python rag/query.py "Nash equilibrium existence" --category econ.TH
```

This prints an augmented prompt: a numbered **CONTEXT** block where each chunk is
cited by `[n] arXiv:<id> — <title>`, followed by the **QUESTION** and an
**ANSWER PROMPT** ready to send to any LLM.

## End-to-end answer generation (Claude API)

The full RAG loop — retrieve → augment → **generate** — is wired up via the
Anthropic SDK. Add `--answer` to have Claude synthesize a cited answer grounded
only in the retrieved context:

```bash
export ANTHROPIC_API_KEY=sk-ant-...           # read from env, never stored in code
./.venv/bin/python rag/query.py "auction revenue equivalence" --answer
./.venv/bin/python rag/query.py "Nash equilibrium existence" --answer --k 8
```

Without `--answer`, the command prints the augmented prompt only and needs no
key. With `--answer` it:

1. retrieves top-k chunks from Chroma,
2. sends them as the user turn (plus a grounding `system` prompt) to Claude,
3. prints the **ANSWER** followed by the **SOURCES** list.

Details:
- **Key handling:** `generate_answer()` calls `anthropic.Anthropic()`, which
  reads `ANTHROPIC_API_KEY` from the environment. The key is **never** written
  to the codebase — safe to commit. If the key is missing, the command prints a
  clear message and exits non-zero (no call is made).
- **Model:** `claude-opus-4-8` by default (override with `--model`), adaptive
  thinking, streamed response.
- **Grounding:** the system prompt instructs Claude to answer *only* from the
  retrieved context and to cite each claim with its `[n]` source marker, or say
  so when the context is insufficient.

`generate_answer(query, k, source, category, model)` is also importable to use
the same path programmatically (returns `(answer_text, hits)`).

## Requirements
See `requirements.txt` (`chromadb>=1.5`, `sentence-transformers>=5.5`,
`anthropic>=0.69`). Use the project venv at `.venv/` (Python 3.13); the
embedding model is already cached.
```bash
./.venv/bin/pip install -r rag/requirements.txt
```

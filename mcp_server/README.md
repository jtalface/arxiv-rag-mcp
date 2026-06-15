# arXiv econ.TH MCP Server

An [MCP](https://modelcontextprotocol.io) server (stdio transport, official
Python `mcp` SDK / `FastMCP`) that exposes query and retrieval tools over a
corpus of arXiv **Theoretical Economics (`econ.TH`)** papers (~4046 papers with
metadata + abstracts; full text for the subset whose PDFs have been extracted).

The server reads through the shared data-access layer (`shared/corpus.py`) so it
stays consistent with the rest of the project. Metadata is loaded and cached at
startup; full-text files are read lazily at query time, so papers whose text is
extracted after the server starts become searchable without a restart. Papers
without full text degrade gracefully to their title + abstract.

## Tools

| Tool | Description |
|------|-------------|
| `search_papers(query, max_results=10, category=None, author=None, date_from=None, date_to=None)` | Keyword/relevance search over title + abstract (+ full text when available) using a BM25-lite score with a title boost. Optional filters: `category` (substring match against arXiv categories), `author` (substring match), and an inclusive ISO `date_from`/`date_to` range on the publication date. Returns a ranked list of `{id, title, authors, primary_category, published, score, snippet, abs_url, has_full_text}`. |
| `get_paper(paper_id)` | Full metadata for one paper. Accepts a bare id (`2401.01234`) or versioned id (`2401.01234v2`). |
| `get_full_text(paper_id, max_chars=20000)` | Extracted full text if available (truncated to `max_chars` with a note); otherwise a clear message plus the abstract as fallback. |
| `list_categories()` | Distinct arXiv categories in the corpus with paper counts, sorted descending. |
| `get_corpus_stats()` | Totals: number of papers, number with full text, publication date range, top authors and categories, plus the on-disk `manifest.json` summary. |
| `get_recent_papers(n=10, category=None)` | Most recently published papers, newest first; optional category filter. |

## Running the server

Use the project virtualenv (Python 3.13). The `mcp` package is already
installed there.

```bash
"/Users/josealface/AI Projects/Claude/my-projects/arxiv/.venv/bin/python" \
  "/Users/josealface/AI Projects/Claude/my-projects/arxiv/mcp_server/server.py"
```

The server speaks JSON-RPC over **stdio**, so running it directly will appear to
"hang" — that is expected; it is waiting for an MCP client. Register it with a
client instead (below).

### Dependencies

```
mcp>=1.27
```

Already installed in the project `.venv`. (See `requirements.txt`.)

## Smoke test

A self-contained test calls each tool's underlying function directly with real
data and exits non-zero on failure:

```bash
cd "/Users/josealface/AI Projects/Claude/my-projects/arxiv" && \
  ./.venv/bin/python mcp_server/test_server.py
```

## MCP client configuration

Add this to your MCP client config (e.g. Claude Desktop's
`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "arxiv-econ-th": {
      "command": "/Users/josealface/AI Projects/Claude/my-projects/arxiv/.venv/bin/python",
      "args": ["/Users/josealface/AI Projects/Claude/my-projects/arxiv/mcp_server/server.py"]
    }
  }
}
```

Or via the Claude Code CLI:

```bash
claude mcp add arxiv-econ-th \
  "/Users/josealface/AI Projects/Claude/my-projects/arxiv/.venv/bin/python" \
  "/Users/josealface/AI Projects/Claude/my-projects/arxiv/mcp_server/server.py"
```

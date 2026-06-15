"""
RAG query CLI for the econ.TH corpus.

Retrieves the top-k chunks for a natural-language question and assembles a
Retrieval-Augmented Generation prompt: a CONTEXT block of cited chunks followed
by the user's QUESTION and an ANSWER PROMPT, ready to send to any LLM.

By default the command prints the assembled augmented context (no API key
needed). Pass --answer to generate a final answer with the Claude API: the
retrieved context is sent to Claude (Anthropic SDK), which synthesizes a cited
answer grounded only in the corpus.

The API key is read from the ANTHROPIC_API_KEY environment variable — it is
never stored in this codebase. Set it before using --answer:
    export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python rag/query.py "mechanism design with incomplete information"
    python rag/query.py "auction revenue" --k 8 --source fulltext
    python rag/query.py "Nash equilibrium existence" --answer
"""
from __future__ import annotations

import argparse
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from retriever import retrieve, Hit  # noqa: E402


def build_context(hits: list[Hit]) -> str:
    """Assemble retrieved chunks into a numbered, cited context block."""
    if not hits:
        return "(no relevant context retrieved)"
    blocks = []
    for i, h in enumerate(hits, 1):
        citation = f"[{i}] arXiv:{h.paper_id} — {h.title}".rstrip(" —")
        authors = h.metadata.get("authors", "")
        meta_line = f"    source={h.metadata.get('source', '')}, " \
                    f"category={h.metadata.get('primary_category', '')}, " \
                    f"score={h.score:.3f}"
        if authors:
            citation += f" ({authors})"
        blocks.append(f"{citation}\n{meta_line}\n{h.text.strip()}")
    return "\n\n".join(blocks)


# System prompt: the assistant's role and grounding rules. Sent as the API
# `system` parameter; kept stable so it caches well across requests.
SYSTEM_PROMPT = (
    "You are a research assistant answering questions about theoretical "
    "economics using a corpus of arXiv econ.TH papers. Answer using ONLY the "
    "retrieved context provided in the user message. Cite every claim with the "
    "source's [n] marker (which maps to an arXiv id). If the context is "
    "insufficient to answer, say so explicitly rather than guessing. Be precise "
    "and concise."
)

DEFAULT_MODEL = "claude-opus-4-8"


def build_user_content(query: str, context: str) -> str:
    """The user turn: retrieved context followed by the question."""
    return (
        "===== CONTEXT =====\n"
        f"{context}\n\n"
        "===== QUESTION =====\n"
        f"{query}\n\n"
        "Using only the context above, answer the question and cite the "
        "supporting [n] sources."
    )


def augment(query: str, k: int = 5, source: str | None = None,
            category: str | None = None) -> tuple[str, list[Hit]]:
    """Retrieve top-k chunks and return (augmented_prompt, hits).

    The returned string is a complete, human-readable RAG prompt: the system
    instructions + CONTEXT (cited) + QUESTION. Useful for inspection and for
    feeding to any LLM. ``generate_answer`` sends the same content to Claude.
    """
    hits = retrieve(query, k=k, source=source, category=category)
    context = build_context(hits)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{build_user_content(query, context)}"
    )
    return prompt, hits


def generate_answer(query: str, k: int = 5, source: str | None = None,
                    category: str | None = None,
                    model: str = DEFAULT_MODEL) -> tuple[str, list[Hit]]:
    """Generate a final, cited RAG answer with the Claude API.

    Retrieves context, sends it to Claude (Anthropic SDK), and returns
    (answer_text, hits). The API key is read from the ANTHROPIC_API_KEY
    environment variable by ``anthropic.Anthropic()`` — it is never hardcoded.

    Streaming is used so large answers don't hit request timeouts; adaptive
    thinking lets the model reason as needed over the retrieved context.
    """
    import os

    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "The 'anthropic' package is required for --answer. "
            "Install it: ./.venv/bin/python -m pip install anthropic"
        ) from e

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export your key first:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-...\n"
            "(The key is read from the environment and never stored in code.)"
        )

    hits = retrieve(query, k=k, source=source, category=category)
    context = build_context(hits)
    user_content = build_user_content(query, context)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    with client.messages.stream(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        message = stream.get_final_message()

    answer = "".join(
        block.text for block in message.content if block.type == "text"
    ).strip()
    return answer, hits


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG query over the econ.TH corpus.")
    ap.add_argument("query", help="Natural-language question.")
    ap.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve.")
    ap.add_argument("--source", choices=["fulltext", "abstract"], default=None,
                    help="Filter by chunk source.")
    ap.add_argument("--category", default=None,
                    help="Filter by primary_category (e.g. econ.TH).")
    ap.add_argument("--answer", action="store_true",
                    help="Generate a final answer with the Claude API "
                         "(requires ANTHROPIC_API_KEY in the environment).")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Claude model id (default: {DEFAULT_MODEL}).")
    args = ap.parse_args()

    if args.answer:
        try:
            answer, hits = generate_answer(
                args.query, k=args.k, source=args.source,
                category=args.category, model=args.model,
            )
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print("===== ANSWER =====")
        print(answer)
        print("\n===== SOURCES =====")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] arXiv:{h.paper_id} — {h.title}")
        return

    prompt, hits = augment(args.query, k=args.k, source=args.source,
                           category=args.category)
    print(prompt)
    print("\n" + "=" * 60)
    print(f"Retrieved {len(hits)} chunks from "
          f"{len({h.paper_id for h in hits})} papers.")


if __name__ == "__main__":
    main()

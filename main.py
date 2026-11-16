"""CLI for the agentic RAG system.

    python main.py ingest [docs_dir]      build / rebuild the index
    python main.py search "query"         test retrieval only (no LLM call)
    python main.py ask "question"         one-shot question
    python main.py chat                   interactive multi-turn chat
"""

from __future__ import annotations

import argparse
import sys

import anthropic

from rag.store import KnowledgeBase


def main() -> None:
    p = argparse.ArgumentParser(description="Agentic RAG with Claude")
    p.add_argument("--index", default=".rag_index", help="index directory (default: .rag_index)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest"); s.add_argument("docs_dir", nargs="?", default="docs")
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=5)
    s = sub.add_parser("ask"); s.add_argument("question"); s.add_argument("-q", "--quiet", action="store_true")
    s = sub.add_parser("chat"); s.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    kb = KnowledgeBase(args.index)

    if args.cmd == "ingest":
        n = kb.ingest(args.docs_dir)
        print(f"Indexed {n} chunks from {len(kb.sources())} documents in '{args.docs_dir}'.")
        return

    if args.cmd == "search":
        for chunk, score in kb.search(args.query, k=args.k):
            print(f"--- [{chunk.source}#{chunk.index}] score={score:.4f}\n{chunk.text}\n")
        return

    from rag.agent import RagAgent

    agent = RagAgent(kb, verbose=not args.quiet)
    try:
        if args.cmd == "ask":
            print(agent.ask(args.question))
            return
        print("Agentic RAG chat. Commands: /reset, /quit")
        while True:
            try:
                q = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q in ("/quit", "/exit"):
                break
            if q == "/reset":
                agent.reset(); print("(history cleared)"); continue
            print(f"\nclaude> {agent.ask(q)}")
    except anthropic.AuthenticationError:
        sys.exit("Authentication failed: set ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        sys.exit("Rate limited by the API; try again shortly.")
    except anthropic.APIStatusError as e:
        sys.exit(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        sys.exit("Could not reach the Anthropic API.")


if __name__ == "__main__":
    main()

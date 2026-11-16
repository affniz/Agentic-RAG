"""CLI for the agentic RAG system.

    python main.py ingest [docs_dir] [--rebuild]   sync the index with a folder (only changed files are re-embedded)
    python main.py search "query" [--mode ...]     test retrieval only (no API call)
    python main.py ask "question"                  one-shot question
    python main.py chat                            interactive multi-turn chat
    python main.py eval                            retrieval metrics (free)
    python main.py eval --answers                  + end-to-end answer grading (calls the API)
"""

from __future__ import annotations

import argparse
import sys

import anthropic

from rag.store import SEARCH_MODES, KnowledgeBase


def print_event(event: dict) -> None:
    if event["type"] == "tool_call":
        shown = ", ".join(f"{k}={v!r}" for k, v in event["args"].items())
        print(f"  -> {event['name']}({shown})")
    elif event["type"] == "usage":
        u = event["usage"]
        cost = f"${event['cost_usd']:.4f}" if event["cost_usd"] is not None else "n/a"
        print(f"  [{u.api_calls} API calls, {u.input_tokens + u.cache_read_tokens + u.cache_write_tokens} in / "
              f"{u.output_tokens} out tokens, ~{cost}]")


def main() -> None:
    p = argparse.ArgumentParser(description="Agentic RAG with Claude")
    p.add_argument("--index", default=".rag_index", help="index directory (default: .rag_index)")
    p.add_argument("--no-rerank", action="store_true", help="disable the cross-encoder reranker")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest")
    s.add_argument("docs_dir", nargs="?", default="docs")
    s.add_argument("--rebuild", action="store_true", help="re-embed everything")

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--mode", choices=SEARCH_MODES, default="hybrid")

    for name in ("ask", "chat"):
        s = sub.add_parser(name)
        if name == "ask":
            s.add_argument("question")
        s.add_argument("-q", "--quiet", action="store_true", help="hide tool calls and cost")
        s.add_argument("--model", default=None, help="e.g. claude-sonnet-5-5, claude-opus-5-5, claude-haiku-5-5")
        s.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])

    s = sub.add_parser("eval")
    s.add_argument("--questions", default="evals/questions.jsonl")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--answers", action="store_true", help="also run the agent and grade answers (costs money)")
    s.add_argument("--model", default=None)
    s.add_argument("--effort", default="medium")
    s.add_argument("--limit", type=int, default=None, help="only the first N questions")

    args = p.parse_args()
    kb = KnowledgeBase(args.index, rerank=not args.no_rerank)

    if args.cmd == "ingest":
        r = kb.ingest(args.docs_dir, rebuild=args.rebuild)
        print(f"Index: {r.chunks} chunks from {len(kb.sources())} documents "
              f"(added {len(r.added)}, updated {len(r.updated)}, removed {len(r.removed)}, unchanged {r.unchanged}).")
        return

    if args.cmd == "search":
        for chunk, score in kb.search(args.query, k=args.k, mode=args.mode):
            print(f"--- {chunk.label()} score={score:.4f}\n{chunk.text}\n")
        return

    from rag.agent import MODEL, RagAgent

    model = getattr(args, "model", None) or MODEL
    try:
        if args.cmd == "eval":
            from rag.evaluate import answer_eval, load_questions, retrieval_eval

            questions = load_questions(args.questions)[: args.limit]
            print(f"Retrieval (k={args.k}, {sum(1 for q in questions if q.get('evidence'))} questions with evidence):")
            for name, m in retrieval_eval(kb, questions, k=args.k).items():
                extra = f"  ({m['note']})" if "note" in m else ""
                print(f"  {name:<15} recall@k={m['recall@k']:.2f}  hit@k={m['hit@k']:.2f}  mrr={m['mrr']:.2f}{extra}")
            if args.answers:
                print(f"\nAnswers ({model}, effort={args.effort}):")
                s = answer_eval(kb, questions, model=model, effort=args.effort)
                print(f"\n  correct={s['correct']:.0%}  partial={s['partially_correct']:.0%}  "
                      f"cited expected sources={s['cited_expected_sources']:.0%}")
                print(f"  avg tool calls={s['avg_tool_calls']:.1f}  avg time={s['avg_seconds']:.1f}s")
                print(f"  cost: agent ${s['agent_cost_usd'] or 0:.3f} + judge ${s['judge_cost_usd'] or 0:.3f}")
                print(f"  details: {s['results_file']}")
            return

        agent = RagAgent(kb, model=model, effort=args.effort, on_event=None if args.quiet else print_event)
        if args.cmd == "ask":
            print(agent.ask(args.question).text)
            return
        print(f"Agentic RAG chat ({model}). Commands: /reset, /cost, /quit")
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
                agent.reset()
                print("(history cleared)")
                continue
            if q == "/cost":
                c = agent.total.cost(model)
                print(f"(session: {agent.total.api_calls} API calls, ~${c or 0:.4f})")
                continue
            print(f"\nclaude> {agent.ask(q).text}")
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

"""CLI for the agentic RAG system.

    python main.py ingest [docs_dir] [--rebuild]   sync the index with a folder (only changed files are re-embedded)
    python main.py search "query" [--mode ...]     test retrieval only (no API call)
    python main.py ask "question"                  one-shot question
    python main.py chat                            interactive multi-turn chat
    python main.py eval [--set evals/hard]         retrieval metrics (free)
    python main.py eval --set evals/hard --answers --systems agentic,naive
                                                   + answer grading, agent vs one-shot RAG (calls the API)
"""

from __future__ import annotations

import argparse
import sys

import anthropic

from rag.store import SEARCH_MODES, KnowledgeBase


def print_event(event: dict) -> None:
    if event["type"] == "model_call":
        retried = f", {event['retries']} retries" if event.get("retries") else ""
        print(f"  [model call {event['step']}: {event['seconds']:.1f}s - waited {event['wait_seconds']:.1f}s "
              f"for the first byte{retried}, {event['output_tokens']} output tokens]")
    elif event["type"] == "tool_call":
        shown = ", ".join(f"{k}={v!r}" for k, v in event["args"].items())
        print(f"  -> {event['name']}({shown})")
    elif event["type"] == "tool_result":
        print(f"     {event['name']} done in {event['seconds'] * 1000:.0f} ms")
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
    s.add_argument("--ocr", action="store_true",
                   help="transcribe scanned PDFs (no text layer) with Claude - sends those PDFs to the API")

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
        s.add_argument("--max-cost", type=float, default=None,
                       help="USD per question; past it the agent stops searching and answers")

    s = sub.add_parser("eval")
    s.add_argument("--set", default="evals/basic", help="eval set directory (evals/basic, evals/hard)")
    s.add_argument("-k", type=int, default=5, help="k for the retrieval metrics")
    s.add_argument("--answers", action="store_true", help="also answer and grade every question (costs money)")
    s.add_argument("--systems", default="agentic", help="comma-separated: agentic,naive")
    s.add_argument("--naive-k", type=int, default=5, help="chunks the naive baseline retrieves")
    s.add_argument("--model", default=None)
    s.add_argument("--effort", default="medium")
    s.add_argument("--judge-model", default=None, help="default claude-sonnet-5-5")
    s.add_argument("--workers", type=int, default=4, help="questions answered in parallel")
    s.add_argument("--category", default=None, help="only questions in this category")
    s.add_argument("--limit", type=int, default=None, help="only the first N questions")
    s.add_argument("--repeats", type=int, default=1, help="run everything N times; report mean, range, unstable questions")

    args = p.parse_args()
    ocr = None
    if getattr(args, "ocr", False):
        from rag.ocr import claude_pdf_transcriber

        ocr = claude_pdf_transcriber()
    kb = KnowledgeBase(args.index, rerank=not args.no_rerank, pdf_ocr=ocr)

    if args.cmd == "ingest":
        try:
            r = kb.ingest(args.docs_dir, rebuild=args.rebuild)
        except FileNotFoundError as e:
            sys.exit(f"{e} - nothing was changed.")
        print(f"Index: {r.chunks} chunks from {len(kb.sources())} documents "
              f"(added {len(r.added)}, updated {len(r.updated)}, removed {len(r.removed)}, unchanged {r.unchanged}; {r.embedded} chunks embedded).")
        for name, reason in r.failed.items():
            print(f"Skipped {name}: {reason}")
        for name in r.empty:
            hint = "" if args.ocr or not name.lower().endswith(".pdf") else " Scanned PDF? Re-run with --ocr."
            print(f"Warning: no text found in {name} - it isn't searchable.{hint}")
        if kb.reranker is not None:
            print("Loading reranker (downloads ~90 MB the first time)...")
            print("Reranker ready." if kb.reranker.warm() else "Reranker unavailable; search will run without it.")
        return

    if args.cmd == "search":
        for chunk, score in kb.search(args.query, k=args.k, mode=args.mode):
            print(f"--- {chunk.label()} score={score:.4f}\n{chunk.text}\n")
        return

    from rag.agent import MODEL, RagAgent

    model = getattr(args, "model", None) or MODEL
    try:
        if args.cmd == "eval":
            run_eval(args, model)
            return

        agent = RagAgent(kb, model=model, effort=args.effort, on_event=None if args.quiet else print_event,
                         max_cost_usd=args.max_cost)
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


def run_eval(args: argparse.Namespace, model: str) -> None:
    from rag.evaluate import JUDGE_MODEL, answer_eval, load_set, retrieval_eval, save_retrieval_run, set_knowledge_base

    es = load_set(args.set)
    questions = [q for q in es.questions if not args.category or q.get("category") == args.category][: args.limit]
    kb = set_knowledge_base(es, rerank=not args.no_rerank)
    print(f"Eval set '{es.name}': {len(questions)} questions, {len(kb.sources())} documents, "
          f"{len(kb.chunks)} chunks{f', today={es.today}' if es.today else ''}")

    print(f"\nSingle-query retrieval with the question text (k={args.k}):")
    metrics = retrieval_eval(kb, questions, k=args.k)
    save_retrieval_run(es.name, args.k, metrics, len(questions))
    for name, m in metrics.items():
        extra = f"  ({m['note']})" if "note" in m else ""
        print(f"  {name:<15} recall@k={m['recall@k']:.2f}  hit@k={m['hit@k']:.2f}  mrr={m['mrr']:.2f}{extra}")
    if not args.answers:
        return

    systems = tuple(x.strip() for x in args.systems.split(",") if x.strip())
    print(f"\nAnswers ({model}, effort={args.effort}, systems={','.join(systems)}):")
    s = answer_eval(kb, questions, model=model, effort=args.effort, systems=systems, today=es.today,
                    naive_k=args.naive_k, judge_model=args.judge_model or JUDGE_MODEL,
                    workers=args.workers, label=es.name, set_name=es.name, repeats=args.repeats)

    res = s["systems"]
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"

    def ranged(stats: dict, key: str) -> str:
        """'95%', or '95% (93-98)' when repeated runs disagree."""
        v, rng = stats.get(key), stats.get("range")
        if isinstance(rng, dict):
            rng = rng.get(key)
        if v is None or not rng or rng[0] == rng[1]:
            return pct(v)
        return f"{v:.0%} ({rng[0] * 100:.0f}-{rng[1] * 100:.0f})"
    w = 22 if s["repeats"] > 1 else 16
    head = "".join(f"{(name if name != 'naive' else f'naive (k={args.naive_k})'):>{w}}" for name in systems)
    print(f"\n{'':<28}{head}")
    if s["repeats"] > 1:
        print(f"{'(mean over ' + str(s['repeats']) + ' runs, range)':<28}")
    print(f"{'correct':<28}" + "".join(f"{ranged(res[n], 'correct'):>{w}}" for n in systems))
    cats = sorted({c for n in systems for c in res[n]['by_category']})
    for c in cats:
        cnt = next(res[n]['by_category'][c]['n'] for n in systems if c in res[n]['by_category'])
        print(f"{'  ' + c + f' ({cnt})':<28}" + "".join(
            f"{ranged(res[n]['by_category'].get(c, {}), 'correct'):>{w}}" for n in systems))
    print(f"{'partially correct':<28}" + "".join(f"{ranged(res[n], 'partially_correct'):>{w}}" for n in systems))
    print(f"{'evidence retrieved':<28}" + "".join(f"{ranged(res[n], 'evidence_recall'):>{w}}" for n in systems))
    print(f"{'cited expected sources':<28}" + "".join(f"{ranged(res[n], 'cited_expected_sources'):>{w}}" for n in systems))
    print(f"{'cited unseen chunks':<28}" + "".join(f"{ranged(res[n], 'cited_unseen'):>{w}}" for n in systems))
    print(f"{'avg API calls':<28}" + "".join(f"{res[n]['avg_api_calls']:>{w}.1f}" for n in systems))
    print(f"{'avg seconds':<28}" + "".join(f"{res[n]['avg_seconds']:>{w}.1f}" for n in systems))
    cost_label = "cost per run (USD)" if s["repeats"] > 1 else "cost (USD)"
    print(f"{cost_label:<28}" + "".join(f"{'$' + format(res[n]['cost_usd'], '.3f'):>{w}}" for n in systems))
    if any(res[n]["errors"] for n in systems):
        print(f"{'errors (left out above)':<28}" + "".join(f"{res[n]['errors']:>{w}}" for n in systems))
    if s["repeats"] > 1:
        for n in systems:
            unstable = res[n]["unstable"]
            print(f"\nunstable for {n} (verdict changed between runs): {', '.join(unstable) if unstable else 'none'}")
    judge_cost = sum(res[n]["judge_cost_usd"] for n in systems)
    print(f"\njudge ({s['judge_model']}) cost ${judge_cost:.3f}. Details: {s['results_file']}")


if __name__ == "__main__":
    main()

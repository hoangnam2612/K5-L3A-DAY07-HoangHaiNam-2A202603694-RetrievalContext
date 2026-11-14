"""Run the real agent over benchmark cases and save full traces.

Gold (data/questions.json) is read ONLY after a run, to compute diagnostics; it is never given to the agent.

Examples:
  python -X utf8 scripts/run_agent.py --name baseline --methods dense --split dev
  python -X utf8 scripts/run_agent.py --name multi --methods dense bm25 hybrid --split dev --ids d01 d25
`--split test` is refused unless --final is given (config must be frozen on dev first).
"""
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from baseline import hit  # noqa: E402
from src.agent import Agent, AgentConfig  # noqa: E402
from src.retrievers import Bm25Retriever, HybridRetriever, load_chunks  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--final", action="store_true")
    p.add_argument("--name", required=True)
    p.add_argument("--methods", nargs="+", default=["dense"], choices=("dense", "bm25", "hybrid"))
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--budget", type=int, default=3500)
    p.add_argument("--max-searches", type=int, default=3)
    p.add_argument("--ids", nargs="*", help="only these case IDs (smoke test)")
    p.add_argument("--no-filter", action="store_true")
    args = p.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("Refusing to run test without --final: freeze the config on dev first.")
    load_dotenv(ROOT / ".env")

    from openai import OpenAI
    cases = [c for c in json.loads((ROOT / "data/agent_cases.json").read_text(encoding="utf-8"))
             if c["split"] == args.split and (not args.ids or c["id"] in args.ids)]
    gold = {q["id"]: q["gold"] for q in json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))}

    retrievers = {"bm25": Bm25Retriever(load_chunks())}
    if {"dense", "hybrid"} & set(args.methods):
        from src.dense import DenseRetriever
        retrievers["dense"] = DenseRetriever()
        retrievers["hybrid"] = HybridRetriever(retrievers["dense"], retrievers["bm25"])
    cfg = AgentConfig(model=args.model, methods=tuple(args.methods), top_k=args.top_k,
                      budget_chars=args.budget, max_searches=args.max_searches, name=args.name)
    agent = Agent(OpenAI(), retrievers, cfg)

    traces = []
    for c in cases:
        flt = None if args.no_filter else c.get("filter")
        t = agent.run(c["id"], c["question"], filters=flt)
        t.update(split=c["split"], answerable=c["answerable"])
        if c["answerable"]:                          # post-hoc diagnostics only
            g = gold[c["id"]]
            ctx_hit = False
            retrieved_hit = False
            chunks_by_id = {ch["id"]: ch for ch in load_chunks()}
            if any(e["event"] == "search" for e in t["events"]):
                final_ids = [e for e in t["events"] if e["event"] == "search"][-1]["context_ids"]
                ctx_hit = any(hit(chunks_by_id[i], g) for i in final_ids)
                retrieved_hit = any(hit(chunks_by_id[r["id"]], g) for e in t["events"] if e["event"] == "search" for r in e["results"])
            t["diagnostics"] = {"context_has_evidence": ctx_hit, "any_search_hit": retrieved_hit}
        traces.append(t)
        print(f"{t['query_id']:5} {t['status']:8} searches={t['n_searches']} cites={t['citations']} "
              f"{t['elapsed_ms']:.0f}ms ${t['cost_usd']:.5f} {t.get('error','')}")

    ans = [t for t in traces if t["answerable"]]
    summary = {"name": args.name, "split": args.split, "config": traces[0]["config"] if traces else None,
               "cases": len(traces), "status": {s: sum(t["status"] == s for t in traces) for s in ("answered", "abstain", "error")},
               "answerable": len(ans),
               "context_evidence_rate": f"{sum(t['diagnostics']['context_has_evidence'] for t in ans)}/{len(ans)}",
               "searches": {str(k): sum(t['n_searches'] == k for t in traces) for k in sorted({t['n_searches'] for t in traces})},
               "tokens": {"prompt": sum(t["usage"]["prompt_tokens"] for t in traces),
                          "completion": sum(t["usage"]["completion_tokens"] for t in traces)},
               "cost_usd": round(sum(t["cost_usd"] for t in traces), 5),
               "median_ms": sorted(t["elapsed_ms"] for t in traces)[len(traces) // 2] if traces else None}
    out = ROOT / "results" / "agent" / f"{args.name}-{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

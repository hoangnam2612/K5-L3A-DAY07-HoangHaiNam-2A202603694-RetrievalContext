"""Compare retrieval methods on the same chunks/questions: Hit@1 and Evidence Hit@3.

Uses the ORIGINAL question as the query. The only filter used is the one stored in the case
(user-provided context); gold is used only to score, never passed to a retriever.

Examples (from repo root, venv active):
  python -X utf8 scripts/eval_retrieval.py --split dev --methods dense bm25 hybrid
  python -X utf8 scripts/eval_retrieval.py --split dev --methods bm25 --bm25-mode both --name bm25-both
`--split test` is refused unless --final is given (config must be frozen first).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from baseline import hit  # noqa: E402  (same hit definition as the starter)
from src.retrievers import Bm25Retriever, HybridRetriever, load_chunks  # noqa: E402


def build(method: str, args, dense=None):
    lex = Bm25Retriever(load_chunks(), mode=args.bm25_mode, fold=args.bm25_fold, k1=args.k1, b=args.b)
    if method == "bm25":
        return lex
    from src.dense import DenseRetriever
    dense = dense or DenseRetriever()
    if method == "dense":
        return dense
    if method == "hybrid":
        return HybridRetriever(dense, lex, pool=args.pool, k=args.rrf_k)
    raise SystemExit(f"unknown method {method}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--final", action="store_true", help="required for --split test (config frozen)")
    p.add_argument("--methods", nargs="+", default=["dense", "bm25", "hybrid"])
    p.add_argument("--name", default=None, help="output folder name (default: method)")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--bm25-mode", default="syllable", choices=("syllable", "bigram", "both"))
    p.add_argument("--bm25-fold", action="store_true", help="strip diacritics")
    p.add_argument("--k1", type=float, default=1.2)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--pool", type=int, default=20)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--no-filter", action="store_true", help="ignore case filters (ablation)")
    args = p.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("Refusing to run test without --final: freeze the config on dev first.")

    questions = [q for q in json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))
                 if q["split"] == args.split]
    chunks = load_chunks()
    dense_shared = None
    rows = []
    for method in args.methods:
        if method in ("dense", "hybrid") and dense_shared is None:
            from src.dense import DenseRetriever
            dense_shared = DenseRetriever()
        r = build(method, args, dense_shared)
        traces, times = [], []
        for q in questions:
            flt = None if args.no_filter else q.get("filter")
            t0 = time.perf_counter()
            res = r.search(q["question"], filters=flt, top_k=args.top_k)
            ms = (time.perf_counter() - t0) * 1000
            times.append(ms)
            marked = [{**{k: v for k, v in c.items() if k != "text"}, "hit": hit(c, q["gold"])} for c in res]
            # Evidence Hit@3 uses the top-3 even if the method retrieved more.
            top3 = marked[:3]
            traces.append({"query_id": q["id"], "question": q["question"], "method": method, "filter": flt,
                           "results": marked, "hit1": bool(top3[:1] and top3[0]["hit"]),
                           "hit3": any(c["hit"] for c in top3), "elapsed_ms": ms,
                           "gold": q["gold"]})
        n = len(traces)
        summary = {"method": method, "split": args.split, "n": n,
                   "hit1": sum(t["hit1"] for t in traces), "hit3": sum(t["hit3"] for t in traces),
                   "hit_at_k": sum(any(c["hit"] for c in t["results"]) for t in traces), "top_k": args.top_k,
                   "median_ms": round(statistics.median(times), 1),
                   "config": {k: getattr(args, k) for k in ("bm25_mode", "bm25_fold", "k1", "b", "pool", "rrf_k", "no_filter")},
                   "missed_hit3": [t["query_id"] for t in traces if not t["hit3"]]}
        out = ROOT / "results" / "retrieval" / f"{args.name or method}-{args.split}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append(summary)
    print(f"{'method':8} {'split':5} {'Hit@1':>8} {'Hit@3':>8} {'Hit@k':>8} {'ms':>8}")
    for s in rows:
        print(f"{s['method']:8} {s['split']:5} {s['hit1']:>3}/{s['n']:<4} {s['hit3']:>3}/{s['n']:<4} "
              f"{s['hit_at_k']:>3}/{s['n']:<4} {s['median_ms']:>8}")
    for s in rows:
        print(s["method"], "miss@3:", ", ".join(s["missed_hit3"]))


if __name__ == "__main__":
    main()

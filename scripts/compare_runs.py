"""Side-by-side automatic stats for agent runs (dev). Automatic only: manual grades are separate."""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
names = sys.argv[1:]
runs = {n: {t["query_id"]: t for t in json.loads((ROOT / f"results/agent/{n}/traces.json").read_text(encoding="utf-8"))} for n in names}


def stats(T):
    a = [t for t in T.values() if t["answerable"]]
    u = [t for t in T.values() if not t["answerable"]]
    meth = collections.Counter(e["method"] for t in T.values() for e in t["events"] if e["event"] == "search")
    return {"answered / answerable": f"{sum(t['status']=='answered' for t in a)}/{len(a)}",
            "context has evidence": f"{sum(t['diagnostics']['context_has_evidence'] for t in a)}/{len(a)}",
            "answered + ctx evidence": sum(t["status"] == "answered" and t["diagnostics"]["context_has_evidence"] for t in a),
            "answered, NO ctx evidence": sum(t["status"] == "answered" and not t["diagnostics"]["context_has_evidence"] for t in a),
            "false abstain (answerable)": sum(t["status"] == "abstain" for t in a),
            "abstain on unanswerable": f"{sum(t['status']=='abstain' for t in u)}/{len(u)}",
            "errors": sum(t["status"] == "error" for t in T.values()),
            "searches/case (mean)": round(sum(t["n_searches"] for t in T.values()) / len(T), 2),
            "tool chosen": dict(meth),
            "cost USD": round(sum(t["cost_usd"] for t in T.values()), 4),
            "median s": round(sorted(t["elapsed_ms"] for t in T.values())[len(T) // 2] / 1000, 1)}


S = {n: stats(r) for n, r in runs.items()}
w = max(len(n) for n in names) + 2
print(f"{'':28}" + "".join(f"{n:>{max(w,22)}}" for n in names))
for k in S[names[0]]:
    print(f"{k:28}" + "".join(f"{str(S[n][k]):>{max(w,22)}}" for n in names))

"""Print a compact reading packet for manual grading: question, expected answer, gold evidence, and for each
config the agent answer + each cited chunk (window around the gold evidence if present, else chunk start).
Usage: python -X utf8 scripts/review_packet.py d01 d02 ... --runs v2-baseline-dense-dev v2-bm25-first-quote-dev"""
import argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("ids", nargs="+"); ap.add_argument("--runs", nargs="+", required=True)
a = ap.parse_args()
C = {c["id"]: c["text"] for c in json.loads((ROOT / "data/chunks.json").read_text(encoding="utf-8"))}
Q = {q["id"]: q for q in json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8"))}
cases = {c["id"]: c for c in json.loads((ROOT / "data/agent_cases.json").read_text(encoding="utf-8"))}
R = {n: {t["query_id"]: t for t in json.loads((ROOT / f"results/agent/{n}/traces.json").read_text(encoding="utf-8"))} for n in a.runs}
for i in a.ids:
    q = Q.get(i); case = cases[i]
    print("=" * 100); print(i, "| CÂU:", case["question"])
    if q: print("   ĐÁP ÁN:", q["expected_answer"], "| EVIDENCE:", q["gold"][0]["evidence"])
    else: print("   (THIẾU EVIDENCE) lý do:", case["absence_reason"])
    for n in a.runs:
        t = R[n][i]; print(f"  -- [{n}] {t['status']} | searches={t['n_searches']} | cites={len(t['citations'])}")
        print("     ANSWER:", t["answer"][:420].replace("\n", " "))
        if t.get("evidence_quote"): print("     QUOTE :", t["evidence_quote"][:200].replace("\n", " "))
        for cid in t["citations"]:
            txt = C[cid]; ev = q["gold"][0]["evidence"] if q else None
            k = txt.find(ev) if ev else -1
            w = txt[max(0, k - 120):k + len(ev) + 120] if k >= 0 else txt[:260]
            print(f"     CITE {cid[-12:]} {'[CÓ gold evidence]' if k >= 0 else '[KHÔNG có gold evidence]'}: ...{w.replace(chr(10), ' ')}...")

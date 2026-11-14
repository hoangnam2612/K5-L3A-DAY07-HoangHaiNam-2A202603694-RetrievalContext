"""Parameter sweep of retrieval methods on DEV only (original query, case filter, same chunks).

Decision rule fixed BEFORE looking at the table: prefer higher Evidence Hit@3, then Hit@1; a non-default
setting must win by >= 2 questions over the default to be adopted (24 questions => 1 question is noise).
Dense query embeddings are computed once and reused, so the sweep costs ~24 embedding calls.
"""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from baseline import hit  # noqa: E402
from src.dense import DenseRetriever  # noqa: E402
from src.retrievers import Bm25Retriever, load_chunks, rrf  # noqa: E402

chunks = load_chunks()
Q = [q for q in json.loads((ROOT / "data/questions.json").read_text(encoding="utf-8")) if q["split"] == "dev"]
dense = DenseRetriever()
POOL = 30
dense_rows = {q["id"]: dense.search(q["question"], filters=q.get("filter"), top_k=POOL) for q in Q}


def score(rank_fn):
    h1 = h3 = hk = 0
    for q in Q:
        res = rank_fn(q)
        h1 += bool(res[:1] and hit(res[0], q["gold"]))
        h3 += any(hit(c, q["gold"]) for c in res[:3])
        hk += any(hit(c, q["gold"]) for c in res[:10])
    return h1, h3, hk


rows = []
# Dense: different cutoffs (same ranking), shows how far evidence sits below top-3
for k in (1, 3, 5, 10, 30):
    hk = sum(any(hit(c, q["gold"]) for c in dense_rows[q["id"]][:k]) for q in Q)
    rows.append({"method": "dense", "params": f"Hit@{k}", "hit1": None, "hit3": None, "hit@10": None, "hit_at_k": f"{hk}/{len(Q)}"})
d1, d3, d10 = score(lambda q: dense_rows[q["id"]])
rows.append({"method": "dense", "params": "default", "hit1": d1, "hit3": d3, "hit@10": d10})
# dense without the case filter (ablation, to see whether the filter matters)
nf = {q["id"]: dense.search(q["question"], top_k=10) for q in Q if q.get("filter")}
rows.append({"method": "dense", "params": "no-filter on filter-cases only", "hit1": None, "hit3": sum(any(hit(c, q["gold"]) for c in nf[q["id"]][:3]) for q in Q if q.get("filter")), "hit@10": None, "n": len(nf)})

for mode, fold, k1, b in itertools.product(("syllable", "bigram", "both"), (False, True), (0.9, 1.2, 1.5, 2.0), (0.4, 0.75, 0.9)):
    bm = Bm25Retriever(chunks, mode=mode, fold=fold, k1=k1, b=b)
    cache = {q["id"]: bm.search(q["question"], filters=q.get("filter"), top_k=POOL) for q in Q}
    h1, h3, h10 = score(lambda q: cache[q["id"]])
    rows.append({"method": "bm25", "params": f"mode={mode} fold={fold} k1={k1} b={b}", "hit1": h1, "hit3": h3, "hit@10": h10})
    if k1 == 1.2 and b == 0.75:       # hybrid sweep only on default-BM25 variants (keeps the grid small)
        for pool, k in itertools.product((10, 20, 30), (20, 60)):
            hy = lambda q: [r for r, _ in rrf([dense_rows[q["id"]][:pool], cache[q["id"]][:pool]], k=k)]
            h1, h3, h10 = score(hy)
            rows.append({"method": "hybrid", "params": f"bm25[{mode},fold={fold}] pool={pool} rrf_k={k}", "hit1": h1, "hit3": h3, "hit@10": h10})

out = ROOT / "results" / "retrieval" / "sweep-dev.json"
out.write_text(json.dumps({"n": len(Q), "rule": __doc__, "rows": rows}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
n = len(Q)
def show(title, sel, top=8):
    print(f"\n== {title}")
    for r in sorted(sel, key=lambda r: (-(r["hit3"] or 0), -(r["hit1"] or 0)))[:top]:
        print(f"  Hit@1 {r['hit1']:>2}/{n}  Hit@3 {r['hit3']:>2}/{n}  Hit@10 {r['hit@10']:>2}/{n}  {r['params']}")
print("dense cutoffs:", [(r['params'], r['hit_at_k']) for r in rows if 'hit_at_k' in r])
print("dense no-filter ablation:", [r for r in rows if 'no-filter' in r['params']])
show("BM25 top", [r for r in rows if r["method"] == "bm25"], 10)
show("BM25 default (syllable, accents, k1=1.2, b=0.75)", [r for r in rows if r["params"] == "mode=syllable fold=False k1=1.2 b=0.75"], 1)
show("Hybrid top", [r for r in rows if r["method"] == "hybrid"], 8)

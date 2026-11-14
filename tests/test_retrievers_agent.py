"""Offline behavioural tests for BM25, RRF and the agent loop. A fake chat client is scripted, so these
test the CODE only; they say nothing about answer quality (see results/ for real runs)."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agent import ABSTAIN_TEXT, Agent, AgentConfig, pack_context, quote_in_chunks  # noqa: E402
from src.retrievers import Bm25Retriever, HybridRetriever, rrf, tokenize  # noqa: E402


def chunk(i, text, period="truoc-cach-mang", page=1):
    return {"id": i, "text": text, "source_id": "namcao", "source": f"{period}/x.md", "page": page,
            "metadata": {"period": period}}


CHUNKS = [chunk("c1", "Chị Chuột bán con chó mực và hai buồng chuối non được bốn hào"),
          chunk("c2", "Ông lý Nhưng bị ba tháng tù án treo", page=2),
          chunk("c3", "Du viết thư báo sẽ về nhà", "sau-cach-mang", 3)]


class Fixed:
    def __init__(self, ids, name="dense"):
        self.ids, self.name, self.calls = ids, name, []
    def search(self, query, *, filters=None, top_k=3):
        self.calls.append((query, filters, top_k))
        by = {c["id"]: c for c in CHUNKS}
        return [{**by[i], "score": 1.0 / (r + 1), "rank": r + 1, "method": self.name} for r, i in enumerate(self.ids)][:top_k]


def call(name, args, cid="t1"):
    return NS(id=cid, function=NS(name=name, arguments=json.dumps(args)))


def reply(*tool_calls):
    return NS(choices=[NS(message=NS(content=None, tool_calls=list(tool_calls)))],
              usage=NS(prompt_tokens=100, completion_tokens=10))


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.chat = NS(completions=NS(create=lambda **kw: next(self.replies)))


class Bm25Tests(unittest.TestCase):
    def test_tokenize_modes_and_fold(self):
        self.assertEqual(tokenize("Chị Chuột", "syllable"), ["chị", "chuột"])
        self.assertEqual(tokenize("Chị Chuột", "both"), ["chị", "chuột", "chị_chuột"])
        self.assertEqual(tokenize("Đòn", fold=True), ["don"])

    def test_ranks_by_keyword_and_respects_filter(self):
        r = Bm25Retriever(CHUNKS)
        self.assertEqual(r.search("buồng chuối", top_k=3)[0]["id"], "c1")
        self.assertEqual(r.search("thư", filters={"period": "sau-cach-mang"})[0]["id"], "c3")
        self.assertEqual(r.search("thư", filters={"period": "truoc-cach-mang"}), [])   # zero scores never returned
        self.assertEqual([x["rank"] for x in r.search("ông lý chuối thư", top_k=3)], [1, 2, 3])

    def test_rrf_and_hybrid(self):
        a = [{"id": "c1"}, {"id": "c2"}]
        b = [{"id": "c2"}, {"id": "c3"}]
        self.assertEqual([r["id"] for r, _ in rrf([a, b])][0], "c2")     # appears in both lists
        h = HybridRetriever(Fixed(["c1", "c2"]), Fixed(["c2", "c3"], "bm25"))
        out = h.search("q", filters={"period": "x"}, top_k=2)
        self.assertEqual([x["id"] for x in out], ["c2", "c1"])
        self.assertEqual({x["method"] for x in out}, {"hybrid"})


class ContextTests(unittest.TestCase):
    def test_dedup_interleave_and_budget(self):
        calls = [[{**CHUNKS[0], "rank": 1}, {**CHUNKS[1], "rank": 2}], [{**CHUNKS[0], "rank": 1}, {**CHUNKS[2], "rank": 2}]]
        ctx, ids = pack_context(calls, 10_000)
        self.assertEqual(ids, ["c1", "c2", "c3"])
        _, small = pack_context(calls, 120)
        self.assertEqual(small, ["c1"])            # later chunks that do not fit are skipped


class AgentTests(unittest.TestCase):
    def agent(self, replies, ids=("c1", "c2"), **cfg):
        ret = Fixed(list(ids))
        return Agent(FakeClient(replies), {"dense": ret}, AgentConfig(**cfg)), ret

    def test_search_then_cited_answer(self):
        a, ret = self.agent([reply(call("search", {"query": "chuối"})),
                             reply(call("answer", {"enough_evidence": True, "answer": "Bốn hào [c1]", "citations": ["c1"]}))])
        t = a.run("q1", "Bán gì?", filters={"period": "truoc-cach-mang"})
        self.assertEqual((t["status"], t["citations"], t["n_searches"]), ("answered", ["c1"], 1))
        self.assertEqual(ret.calls, [("chuối", {"period": "truoc-cach-mang"}, 3)])   # filter comes from caller only
        self.assertEqual(t["events"][0]["context_ids"], ["c1", "c2"])
        self.assertEqual(t["usage"]["llm_calls"], 2)

    def test_invalid_citation_gets_one_retry_then_abstains(self):
        bad = call("answer", {"enough_evidence": True, "answer": "x [zzz]", "citations": ["zzz"]})
        a, _ = self.agent([reply(call("search", {"query": "q"})), reply(bad), reply(bad)])
        t = a.run("q", "?")
        self.assertEqual((t["status"], t["answer"], t["citations"]), ("abstain", ABSTAIN_TEXT, []))
        self.assertEqual([e["event"] for e in t["events"]].count("invalid_answer"), 2)

    def test_missing_inline_marker_is_appended_not_rejected(self):
        a, _ = self.agent([reply(call("search", {"query": "q"})),
                           reply(call("answer", {"enough_evidence": True, "answer": "Bốn hào", "citations": ["c1"]}))])
        t = a.run("q", "?")
        self.assertEqual((t["status"], t["answer"], t["citations"]), ("answered", "Bốn hào [c1]", ["c1"]))
        self.assertIn("citations_appended", [e["event"] for e in t["events"]])

    def test_model_abstains_and_answer_before_search_is_rejected(self):
        a, _ = self.agent([reply(call("answer", {"enough_evidence": False, "answer": "?", "citations": []})),
                           reply(call("search", {"query": "q"})),
                           reply(call("answer", {"enough_evidence": False, "answer": "", "citations": []}))])
        t = a.run("q", "?")
        self.assertEqual(t["status"], "abstain")
        self.assertEqual(t["n_searches"], 1)       # first answer (no search yet) was bounced

    def test_search_limit_and_provider_error(self):
        s = lambda cid: reply(call("search", {"query": "q"}, cid))
        a, ret = self.agent([s("1"), s("2"), s("3"), s("4"), s("5")], max_searches=2)
        t = a.run("q", "?")
        self.assertEqual(t["status"], "abstain")
        self.assertEqual(len(ret.calls), 2)

        class Boom:
            chat = NS(completions=NS(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("rate limit"))))
        t = Agent(Boom(), {"dense": Fixed(["c1"])}, AgentConfig()).run("q", "?")
        self.assertEqual(t["status"], "error")
        self.assertIn("rate limit", t["error"])    # error is reported, never replaced by a fake result

    def test_multi_method_tool_schema_and_dispatch(self):
        d, l = Fixed(["c1"]), Fixed(["c2"], "bm25")
        a = Agent(FakeClient([reply(call("search", {"query": "q", "method": "bm25"})),
                              reply(call("answer", {"enough_evidence": True, "answer": "[c2]", "citations": ["c2"]}))]),
                  {"dense": d, "bm25": l}, AgentConfig(methods=("dense", "bm25")))
        self.assertIn("method", a.tools[0]["function"]["parameters"]["properties"])
        t = a.run("q", "?")
        self.assertEqual((len(d.calls), len(l.calls), t["events"][0]["method"]), (0, 1, "bm25"))


class QuoteTests(unittest.TestCase):
    def test_quote_verification(self):
        t = ["Chị Chuột bán con chó mực và hai buồng chuối non được bốn hào."]
        self.assertTrue(quote_in_chunks("hai buồng chuối non được bốn hào", t))
        self.assertTrue(quote_in_chunks("Chị  Chuột bán con chó mực ... được bốn hào.", t))    # '...' joins verbatim parts
        self.assertFalse(quote_in_chunks("chị chuột bán con mèo và hai buồng chuối", t))        # paraphrase rejected
        self.assertFalse(quote_in_chunks("", t))
        self.assertFalse(quote_in_chunks("ngắn", t))                                          # too short to prove anything

    def test_agent_rejects_unverifiable_quote_then_abstains(self):
        bad = call("answer", {"enough_evidence": True, "answer": "Bốn hào [c1]", "citations": ["c1"],
                              "evidence_quote": "Chị Chuột nhận được năm hào"})
        good = call("answer", {"enough_evidence": True, "answer": "Bốn hào [c1]", "citations": ["c1"],
                               "evidence_quote": "hai buồng chuối non được bốn hào"})
        a = Agent(FakeClient([reply(call("search", {"query": "q"})), reply(bad), reply(good)]),
                  {"dense": Fixed(["c1"])}, AgentConfig(require_quote=True))
        self.assertIn("evidence_quote", a.tools[1]["function"]["parameters"]["required"])
        t = a.run("q", "?")
        self.assertEqual((t["status"], t["evidence_quote"]), ("answered", "hai buồng chuối non được bốn hào"))
        a2 = Agent(FakeClient([reply(call("search", {"query": "q"})), reply(bad), reply(bad)]),
                   {"dense": Fixed(["c1"])}, AgentConfig(require_quote=True))
        self.assertEqual(a2.run("q", "?")["status"], "abstain")


if __name__ == "__main__":
    unittest.main()

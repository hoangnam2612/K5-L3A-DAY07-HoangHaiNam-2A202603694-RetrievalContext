"""Build the Day 7 benchmark (data/questions.json, data/agent_cases.json) and verify it.

Dev  = 18 Day 6 dev questions + 6 old Day 6 test questions (already seen -> demoted to dev)
       + 3 new unanswerable questions.
Test = NEW questions on stories never used before (5, 11, 36, 46): 8 answerable + 3 unanswerable.
Run: python -I -X utf8 scripts/build_benchmark.py
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SID = "namcao"
chunks = json.loads((ROOT / "data/chunks.json").read_text(encoding="utf-8"))
day6 = json.loads((ROOT / "data/day6/benchmark_day6.json").read_text(encoding="utf-8"))
norm = lambda s: unicodedata.normalize("NFC", s)


def ans(id_, split, question, expected, page, evidence, qtype, flt=None, origin="new"):
    q = {"id": id_, "split": split, "type": qtype, "question": question, "expected_answer": expected,
         "gold": [{"source_id": SID, "page": page, "evidence": evidence}], "origin": origin}
    if flt:
        q["filter"] = flt
    return q


def unans(id_, split, question, reason, keywords, flt=None):
    q = {"id": id_, "split": split, "question": question, "answerable": False,
         "expected_behavior": "search_and_abstain", "absence_reason": reason}
    if flt:
        q["filter"] = flt
    return q, keywords


# ---- dev answerable: Day 6 dev (d01-d18) then demoted Day 6 test (d19-d24) ----
dev = []
for i, x in enumerate([x for x in day6 if x["split"] == "dev"] + [x for x in day6 if x["split"] == "test"], 1):
    origin = f"day6:{x['id']}" + (" (test cũ, đã bị xem -> chuyển sang dev)" if x["split"] == "test" else "")
    dev.append(ans(f"d{i:02d}", "dev", x["question"], x["expected_answer"], x["gold_page"], x["evidence_quote"],
                   x["type"], x.get("filter"), origin))

# ---- new test (answerable) ----
test = [
    ans("tn01", "test", "Để ông lý Nhưng làm ngơ việc vợ chồng nhà kia không có giấy giá thú, anh chồng phải biện cho ông bao nhiêu tiền?",
        "Hai chục bạc.", 36, "biện cho ông lý hai chục bạc", "truc_tiep", {"period": "truoc-cach-mang"}),
    ans("tn02", "test", "Ông khóa Mẫn phát đơn kiện lý trưởng tất cả bao nhiêu khoản?",
        "Mười tám khoản.", 36, "ông phát đơn kiện lý trưởng tất cả mười tám khoản", "truc_tiep"),
    ans("tn03", "test", "Cuối cùng, ông lý Nhưng hả giận với ông khóa Mẫn bằng cách nào, ở đâu?",
        "Ra bãi tha ma, đái lên mộ bố ông khóa Mẫn.", 36, "Chính là mộ bố ông khóa Mẫn", "dien_dat_lai"),
    ans("tn04", "test", "Chiếc áo ba-đơ-xuy mà Trạch Văn Đoành mặc suốt mùa đông mua hết bao nhiêu và mua vào dịp nào?",
        "Bảy mươi quan, mua hồi đi lính sang Tây.", 11, "Hắn mua hồi đi lính sang Tây. Có bảy mươi quan.", "dieu_kien"),
    ans("tn05", "test", "Hai cái móng giò bị mất trong ngày giã đám rốt cuộc ông Cửu Đoành đem cho ai?",
        "Cho anh kép và cô đào, mỗi người một cái.", 11, "Ông rút một tay áo ra, quẳng một cái móng giò cho anh kép.", "dien_dat_lai"),
    ans("tn06", "test", "Vì sao Liễu phải cuốc vườn đến khuya hơn những người khác trong làng?",
        "Vì chị có ba con nhỏ nên phải chạy giặc sớm hơn mọi người, ban ngày làm được ít, phải làm bù bằng cách làm khuya.",
        46, "Liễu những ba con, cùng bé cả. Chị phải chạy sớm hơn mọi người.", "dieu_kien", {"period": "sau-cach-mang"}),
    ans("tn07", "test", "Kiện hình dung sẽ xây gì ở chỗ giáp làng Sơn, và lợi ích ra sao?",
        "Xây một cái cổng, cứu được non một trăm mẫu ruộng.", 46, "Xây một cái cổng ở chỗ giáp làng Sơn, cứu được non một trăm mẫu ruộng", "truc_tiep"),
    ans("tn08", "test", "Khi không còn ra chợ nữa, cô Hoàn nhận làm việc gì để kiếm thêm tiền?",
        "Nhận bế trẻ con thuê.", 5, "cô nhận bế trẻ con thuê", "dien_dat_lai"),
]

# ---- unanswerable: (question, keywords that must NOT reveal an answer; reviewed manually) ----
unanswerable = [
    unans("d25", "dev", "Anh cu chị cu trong truyện Con mèo ở làng nào?",
          "Truyện Con mèo không nêu tên làng của vợ chồng anh cu; cần grep 'anh cu' toàn corpus.", ["anh cu", "chị cu"]),
    unans("d26", "dev", "Con Mực trong truyện Cái chết của con Mực được bao nhiêu tuổi?",
          "Văn bản chỉ nói Mực 'già hơn trong hai con chó', không nêu tuổi.", ["con Mực", "Mực"]),
    unans("d27", "dev", "Vợ chồng Lúng trong truyện Đòn chồng đã lấy nhau được mấy năm?",
          "Cần kiểm tra truyện Đòn chồng (trang 12) không nêu số năm hôn nhân.", ["Lúng"]),
    unans("tn09", "test", "Ông lý Nhưng bị phạt bao nhiêu tiền khi thua kiện?",
          "Truyện chỉ nói 'ba tháng tù án treo; chân lý trưởng thì bị cách', không có khoản tiền phạt.", ["tiền phạt", "phạt vạ", "lý Nhưng"]),
    unans("tn10", "test", "Trạch Văn Đoành có mấy người con?",
          "Truyện chỉ nói hắn về làng 'với một con vợ theo', không nhắc con cái.", ["Đoành"]),
    unans("tn11", "test", "Kiện giữ chức vụ gì trong đơn vị kháng chiến của anh?",
          "Truyện Đợi chờ chỉ nói Kiện 'đi xa' theo đoàn thể, không nêu đơn vị hay chức vụ.", ["Kiện"]),
]

# ---------------- verification ----------------
problems = []
for q in dev + test:
    g = q["gold"][0]
    pg = [c for c in chunks if c["page"] == g["page"]]
    inside = [c["id"] for c in pg if norm(g["evidence"]) in norm(c["text"])]
    anywhere = [c["id"] for c in chunks if norm(g["evidence"]) in norm(c["text"])]
    if len(inside) != 1:
        problems.append(f"{q['id']}: evidence in {len(inside)} chunk(s) of page {g['page']}: {inside}")
    elif set(anywhere) != set(inside):
        problems.append(f"{q['id']}: evidence also appears elsewhere: {anywhere}")
    if q.get("filter"):
        per = {c["metadata"]["period"] for c in pg}
        if per != {q["filter"]["period"]}:
            problems.append(f"{q['id']}: filter {q['filter']} != story period {per}")
print("answerable dev:", len(dev), "test:", len(test), "| problems:", len(problems))
for p in problems:
    print("  PROBLEM", p)

print("\nKeyword scan for unanswerable questions (review each hit manually):")
for q, kws in unanswerable:
    pages = {}
    for k in kws:
        for c in chunks:
            if norm(k).lower() in norm(c["text"]).lower():
                pages.setdefault(c["page"], 0)
                pages[c["page"]] += 1
    print(f"  {q['id']} {q['question']}\n      keyword pages: {pages}")

cases = [{"id": q["id"], "split": q["split"], "question": q["question"], "answerable": True,
          "expected_behavior": "search_and_answer", **({"filter": q["filter"]} if q.get("filter") else {})}
         for q in dev + test]
cases += [u for u, _ in unanswerable]
cases.sort(key=lambda c: (c["split"] != "dev", c["id"]))
(ROOT / "data/questions.json").write_text(json.dumps(dev + test, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(ROOT / "data/agent_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("\nwrote data/questions.json (%d), data/agent_cases.json (%d)" % (len(dev + test), len(cases)))
sys.exit(1 if problems else 0)

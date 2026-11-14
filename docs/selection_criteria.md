# Tiêu chí chọn cấu hình (chốt trên dev, trước khi chạy test)

Viết trước khi xem kết quả của các lượt `v2-bm25-first-quote-r2/r3` và `v2-kw-r1/r2`. Test chưa từng được chạy ở thời điểm viết (`results/` không có thư mục `*-test`).

## Cấu hình ứng viên

| Nhãn | Mô tả (khác biệt so với dòng trên) |
|---|---|
| A `baseline-dense` | Agent chỉ có tool dense (pgvector, text-embedding-3-small), top-3, context 3.500 ký tự, tối đa 3 lượt search |
| B `bm25-first` | A + cho agent 3 tool (bm25/dense/hybrid), BM25 là mặc định và có mô tả mạnh yếu từng tool |
| C `bm25-first-quote` | B + bắt `evidence_quote` chép nguyên văn từ đoạn được trích dẫn, code kiểm tra chuỗi con |
| D `kw` | C + hướng dẫn viết truy vấn bằng 2–4 từ khóa, thử từ đồng nghĩa khi lượt đầu không ra |

`bm25-first-strict` (chỉ thêm lời dặn đối chiếu chi tiết) đã thử và không cải thiện; ghi trong báo cáo như kết quả âm.

## Tiêu chí, theo thứ tự ưu tiên (trên dev, 24 câu có đáp án + 3 câu thiếu evidence)

1. **Context Evidence Rate** (số câu có evidence trong context cuối / 24), lấy trung bình trên các lần chạy lặp.
2. **Điểm grounded trung bình /2 và số câu citation hợp lệ** (chấm tay).
3. **Abstention đúng** (câu thiếu evidence) và **từ chối nhầm** (câu có đáp án).
4. **Chi phí và thời gian** — chỉ để phân thắng bại khi 1–3 hòa.

## Quy tắc quyết định

- Dev chỉ có 24 câu có đáp án, nên chênh **1 câu là nhiễu**. Cấu hình phức tạp hơn chỉ được chọn thay cấu hình đơn giản hơn khi hơn ít nhất **2 câu** ở tiêu chí 1 (trung bình các lần lặp), hoặc hơn rõ ở tiêu chí 2 mà không tệ hơn ở tiêu chí 3.
- Độ nhiễu được đo bằng khoảng min–max của Context Evidence Rate trên các lần chạy lặp cùng cấu hình; chênh lệch nằm trong khoảng này không được coi là cải thiện.
- Một cấu hình chỉ được chấm tay đầy đủ (0/1/2, citation) khi nó đủ tiêu chí để vào vòng cuối.

## Quy trình test

1. Commit mã nguồn và tài liệu này; ghi mã commit vào báo cáo.
2. Chạy baseline A và cấu hình đã chọn **một lần** trên test mới (`--split test --final`), không chạy lại để chọn kết quả đẹp.
3. Chấm tay tối thiểu 5 câu có đáp án + 2 câu thiếu evidence (cả hai cấu hình), kèm điểm context 0/1/2.
4. Nếu xem test để chỉnh bất cứ điều gì, phải tạo test mới và ghi rõ trong báo cáo.

## Giới hạn đã biết

- Nhãn do một người chấm (tác giả), nháp ban đầu do trợ lý AI soạn rồi tác giả xem lại: có nguy cơ thiên lệch.
- Một số hướng dẫn trong prompt (ưu tiên BM25, viết truy vấn từ khóa) xuất phát từ phân tích lỗi trên chính dev, nên kết quả dev có thể lạc quan; test độc lập là phép kiểm tra thật.
- Dev nhỏ (24 câu có đáp án) và chỉ 3 câu thiếu evidence: kết luận về abstention rất yếu.

---

# Quyết định cuối (ghi sau khi xem toàn bộ kết quả dev, TRƯỚC khi chạy test)

Test chưa được chạy ở thời điểm ghi (`results/agent/` không có thư mục `*-test`).

## Số liệu dev dùng để quyết định (24 câu có đáp án + 3 câu thiếu evidence)

| Cấu hình | Số lượt | Context có evidence /24 (TB [min–max]) | Grounded 0/1/2 (chấm tay) | TB /2 | Citation hợp lệ | Từ chối nhầm | Từ chối đúng /3 | $/lượt | Thời gian trung vị |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| A `baseline-dense` | 1 | 9 | 14/1/9 | 0,79 | 17/22 | 10/24 | 1/3 | 0,040 | 3,8 s |
| B `bm25-first` | 1 | 18 | 5/1/18 | 1,54 | 28/32 | 4/24 | 0/3 | 0,038 | 2,9 s |
| C `bm25-first-quote` | 3 | 19,0 [18–20] | 3/1/20 (chỉ lượt r1) | 1,71 | 28/30 | 3/24 (lượt r1); 3–6 qua 3 lượt | 1/3 (câu đúng đổi theo lượt) | 0,044 | 3,4 s |
| D `kw` | 2 | 18,0 [17–19] | không chấm (không vào vòng cuối) | – | – | 3–6 | 0/3 | 0,047 | 3,1 s |
| `strict` (đã thử, giao thức cũ) | 1 | 18 | không chấm | – | – | 7/24 | 0/3 | 0,049 | 4,0 s |

Nhãn chấm tay: `results/reviews/*-dev.json`; số liệu tổng hợp: `results/agent/*-dev/metrics.json`.

## Áp dụng quy tắc đã chốt

- **A bị loại**: kém rõ rệt ở mọi tiêu chí (context 9 so với 18–19).
- **D bị loại**: Context Evidence Rate trung bình 18,0 < C (19,0), không cứu được các câu mất evidence do truy vấn không khớp (d11, d18, d20), làm tụt d09 (C: 3/3, D: 0/2), từ chối đúng 0/3. Giả thuyết bị bác bỏ.
- **B so với C**: C hơn B 1 câu ở tiêu chí 1 (trung bình 19,0 so với 18), nhỏ hơn ngưỡng 2 câu và nằm trong biên độ nhiễu của C (18–20). Ở tiêu chí 2, C hơn B 2 câu điểm 2 (20 so với 18), nhưng đó là lượt r1 — lượt tốt nhất trong 3 lượt của C (context 20/24; r2: 19, r3: 18) — còn B chỉ chạy một lần nên chưa đo được nhiễu riêng. Do đó chưa đủ "hơn rõ" để chọn cấu hình phức tạp hơn.
- **Hòa trong nhiễu → chọn cấu hình đơn giản hơn: B `bm25-first`.** B cũng rẻ hơn khoảng 11% và nhanh hơn khoảng 0,5 giây.

**Cấu hình chốt để chạy test: B `bm25-first`** (cùng baseline A để so sánh):
`python -X utf8 scripts/run_agent.py --name test-bm25-first --methods bm25 dense hybrid --guidance bm25-first --split test --final`
`python -X utf8 scripts/run_agent.py --name test-baseline-dense --methods dense --split test --final`
(model `gpt-4.1-mini`, temperature 0, top-k 3, context 3.500 ký tự, tối đa 3 lượt search, không `--require-quote`, không `--query-guide`.)

## Điều B không giải quyết được (ghi để báo cáo trung thực)

- Từ chối đúng câu thiếu evidence 0/3 trên dev: agent trả lời lạc đề hoặc bịa (d25, d27), hoặc nói "không nêu" nhưng vẫn đánh dấu đã đủ bằng chứng (d26). Chưa có cấu hình nào sửa ổn định điều này (C chỉ 1/3, không ổn định giữa các lượt).
- Mất evidence do BM25 không khớp từ khóa (d11, d18, d20) dẫn đến từ chối nhầm; hybrid/rerank chưa được thử cho vấn đề này.
- Quyết định dựa trên dev nhỏ và nhãn một người chấm; kết quả test (độc lập) mới là phép kiểm tra thật.

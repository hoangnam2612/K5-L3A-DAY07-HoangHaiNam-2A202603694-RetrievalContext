# Code Day 7 — Agentic RAG trên nền Day 6

| File | Công dụng |
|---|---|
| `day6/` | Code Day 6 được chép sang (embedding OpenAI `text-embedding-3-small` 1536 chiều, `PgVectorStore`, chunking, ingestion) để repo tự đủ; nguồn: repo Day 6 |
| `dense.py` | `DenseRetriever`: query embedding + tìm cosine trên pgvector (strategy B, 1.310 chunk), hỗ trợ filter metadata |
| `retrievers.py` | `Bm25Retriever` (BM25 tự viết, tách âm tiết/bigram, tùy chọn bỏ dấu), `rrf`, `HybridRetriever` (RRF dense + BM25) |
| `agent.py` | Agent dùng model chat thật (`gpt-4.1-mini`) với tool `search(query, method)` và `answer(...)`; chọn context theo budget; kiểm tra citation; ghi trace |

Scripts: `scripts/export_chunks.py` (xuất chunk từ pgvector), `scripts/build_benchmark.py` (dựng và kiểm tra benchmark), `scripts/eval_retrieval.py` (Hit@1/Hit@3 theo phương pháp), `scripts/run_agent.py` (chạy agent, lưu trace).

## Chạy lại

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # Windows; Linux/macOS: .venv/bin/python
cp .env.example .env            # điền OPENAI_API_KEY (không commit)
docker compose -f infra/docker-compose.yml -p day7_pgvector up -d --wait           # Postgres 16 + pgvector 0.8.7, cổng 127.0.0.1:55433
python -X utf8 scripts/load_vectors.py --dry-run                   # ước tính chi phí, không gọi API, không ghi DB
python -X utf8 scripts/load_vectors.py                             # tái tạo index từ data/chunks.json (≈ 0,011 USD, ≈ 25 giây)
python -X utf8 -m unittest discover -s tests -v                    # test offline (mock, chỉ kiểm tra code)
python -X utf8 scripts/eval_retrieval.py --split dev --methods dense bm25 hybrid
python -X utf8 scripts/run_agent.py --name baseline-dense --methods dense --split dev
python -X utf8 scripts/run_agent.py --name multi-tool --methods dense bm25 hybrid --split dev
```

`--split test` bị từ chối nếu không có `--final`: chỉ dùng sau khi đã chốt cấu hình trên dev.

## Tái tạo index Day 6 (cho người chấm)

Database Day 6 của tác giả không được nộp. `data/chunks.json` (1.310 chunk strategy B, xuất từ Day 6; nguồn và giấy phép trong `data/day6/manifest.json`) cùng `scripts/load_vectors.py` đủ để dựng lại index với **cùng embedding model** (`text-embedding-3-small`, 1.536 chiều) và cùng schema bảng `chunks` của Day 6. Script an toàn khi chạy lại (upsert theo `(strategy, chunk_id)`, không nhân đôi) và báo lỗi rõ ràng, không dùng vector giả.

Đã kiểm chứng trên một Postgres trống tách riêng: nạp 1.310 dòng (557.213 token, ≈ 0,011 USD), chạy lại vẫn 1.310 dòng. Truy xuất dense trên dev cho Hit@1 = 8/24 và Hit@3 = 10/24 giống index gốc, danh sách câu trượt giống hệt; top-3 trùng cả ID lẫn thứ tự ở 22/24 câu, 2 câu còn lại (d04, d20) chỉ đổi thứ tự giữa các chunk có điểm gần bằng nhau (chênh tối đa 5e-4) vì embedding API không hoàn toàn tất định. Vì vậy các con số retrieval có thể lệch rất nhỏ so với báo cáo khi người chấm tự tái tạo.

## Quyết định thiết kế

- **Filter không do model chọn:** filter `period` lấy từ case (ngữ cảnh người dùng cung cấp), áp dụng trong `search`, và được thông báo cho model trong tin nhắn người dùng. Gold không bao giờ đi vào agent hay retriever.
- **Context:** gộp kết quả các lượt search, loại trùng, xen kẽ theo rank giữa các lượt, cắt theo `budget_chars` (gồm cả header). Chỉ giữ ngữ cảnh tích lũy mới nhất trong hội thoại để budget không bị nhân lên.
- **Citation:** phải là `[chunk-id]` có trong context; sai thì bị từ chối một lần rồi chuyển thành từ chối trả lời. Việc này chỉ kiểm tra ID, **nguồn có hỗ trợ câu trả lời hay không vẫn phải đọc tay**.
- **Lỗi:** lỗi model/DB được ghi `status: "error"` kèm thông báo, không có fallback giả.
- **Giá model** trong `agent.py` (`PRICES`) là giá nhớ từ tài liệu công khai, chưa đối chiếu lại; kiểm tra trước khi đưa chi phí vào báo cáo.

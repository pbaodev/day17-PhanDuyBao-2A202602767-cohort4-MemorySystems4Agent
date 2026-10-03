# Phân tích kết quả — Day 17: Memory Systems for AI Agent

Số liệu lấy từ `python src/benchmark.py` (offline, deterministic; ngưỡng compact 1000 token, giữ 4 message gần nhất; token ước lượng ≈ 4 ký tự/token).

## Kết quả

**Standard Benchmark** (10 hội thoại, user `dungct`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 732 | 13235 | 0.00 | 0.00 | 0 | 0 |
| Advanced | 1469 | 23204 | 1.00 | 1.00 | 399 | 0 |

**Long-Context Stress Benchmark** (1 hội thoại 16 lượt, user `dungct_stress`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 237 | 22029 | 0.00 | 0.00 | 0 | 0 |
| Advanced | 427 | 10984 | 1.00 | 1.00 | 248 | 3 |

## Vì sao Advanced recall tốt hơn Baseline?
Câu hỏi recall được hỏi ở **thread mới**. Baseline chỉ giữ message theo `thread_id` nên thread mới là trang trắng (nó vẫn nhớ *trong* thread, test `test_cross_session_recall` kiểm chứng điều này). Advanced trích fact ổn định (tên, nơi ở, nghề, style, đồ uống/món ăn, thú cưng, mối quan tâm) và ghi vào `User.md`; file này sống ngoài thread (và ngoài process) nên thread nào của cùng user cũng đọc lại được. Correction được xử lý bằng *thay thế*: sau "giờ mình ở Huế chứ không còn ở Đà Nẵng", `User.md` chỉ còn `location: Huế`, không giữ fact cũ sai cạnh fact mới.

## Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?
Ở Standard (mỗi thread ~10 lượt ngắn, chưa chạm ngưỡng nên **0 compaction**), Advanced mất thêm: (a) `User.md` được nhét vào prompt **mọi lượt**, (b) ghi `User.md` và câu xác nhận "Đã ghi nhớ — …" tốn thêm token sinh ra, (c) các thread recall mới phải nạp cả `User.md`. Kết quả: prompt tokens +75%, agent tokens ×2. Với thread ngắn, lịch sử đầy đủ còn rẻ hơn hồ sơ + tóm tắt, nên bộ nhớ dài hạn là chi phí thuần, đổi lại recall 0 → 1.

## Vì sao compact giúp ở hội thoại dài?
Prompt của Baseline là *toàn bộ* lịch sử gửi lại mỗi lượt, nên tổng prompt tokens tăng gần bậc hai theo số lượt. Ở stress (mỗi lượt ~150 token), Advanced vượt ngưỡng 3 lần; mỗi lần nó gom message cũ thành summary ngắn (tối đa 8 dòng) và chỉ giữ 4 message gần nhất, nên prompt mỗi lượt bị chặn trên: **22029 → 10984 (−50%)** mà recall vẫn 1.00 vì các fact quan trọng nằm ở `User.md`, không phụ thuộc summary. Lưu ý compact **chỉ** giảm `prompt tokens processed`; `agent tokens only` của Advanced vẫn cao hơn (ghi memory + xác nhận). Tiết kiệm sẽ lớn hơn khi hội thoại dài hơn nữa (Baseline tăng bậc hai, Advanced gần tuyến tính).

## `User.md` tăng trưởng thế nào, rủi ro gì?
- Tăng trưởng nhỏ và có chặn: 399 B (standard) và 248 B (stress). Scalar fact (tên, nơi ở, nghề…) bị *ghi đè* nên không phình; list fact (`interests`, `style`) bị giới hạn `MAX_LIST_ITEMS = 8`, mục được nhắc lại thì đẩy xuống cuối nên mục cũ nhất rụng trước (một dạng decay theo recency).
- Rủi ro: (1) **lưu sai fact** — lệnh "Nhắc lại style mình thích…" từng bị trích nhầm thành sở thích; đã sửa và có test hồi quy `test_recall_requests_never_pollute_user_md`; (2) correction sai ghi đè fact đúng mà không có lịch sử để khôi phục; (3) rule-based extractor chỉ bắt các mẫu câu đã định nghĩa, câu diễn đạt khác sẽ bị bỏ sót (false negative); (4) summary là heuristic (cắt câu đầu), có thể mất chi tiết — nên fact quan trọng phải ở `User.md`, không dựa vào summary; (5) dữ liệu cá nhân nằm trong file thường, cần cân nhắc quyền truy cập/xoá.

## Bonus đã làm
| Bonus | Giải quyết gì | Rủi ro mới |
|---|---|---|
| **Confidence threshold** (`min_confidence=0.6`): mỗi fact có điểm tin cậy; hedge ("chắc là", "hình như") −0.35, correction ("đính chính", "chứ không") +0.2 | Không ghi "Chắc là mình ở Hà Nội" vào `User.md`; không bỏ lỡ correction | Ngưỡng và trọng số đặt tay; fact đúng nhưng nói dè dặt sẽ bị bỏ |
| **Conflict handling**: scalar fact bị thay thế; mệnh đề phủ định ("không còn ở X", "không còn làm Y") bị loại trước khi trích | Giữ đúng Huế / MLOps engineer thay vì Đà Nẵng / backend | Mất lịch sử thay đổi |
| **Noise guard**: bỏ câu hỏi, câu đùa ("hay là chuyển sang product manager"), câu giả định ("Nếu…"), yêu cầu recall | "Hà Nội", "product manager" không thành fact; hỏi không thành lưu | Có thể bỏ qua câu giả định thật sự là fact |
| **Decay theo recency** cho list fact (cap 8) | Chặn `User.md` phình | Sở thích cũ nhưng thật có thể rụng |

## Giới hạn cần nói thẳng
- Kết quả offline đo **lớp memory**, không đo chất lượng LLM: câu trả lời offline là template sinh từ `User.md`, nên recall 1.00 cho thấy memory layer hoạt động, không dự đoán điểm khi dùng LLM thật. `Response quality` là heuristic (độ phủ fact, hơi phạt câu dài), không phải judge model.
- Extractor và dữ liệu mẫu được chỉnh cùng nhau (rule bắt các mẫu câu của bộ dữ liệu), nên recall 1.00 trên dữ liệu này có thể lạc quan hơn trên văn bản tự do.
- Token là ước lượng ≈ 4 ký tự/token, chỉ để so sánh tương đối.
- Đường live (LangChain `create_agent`, tool đọc/ghi `User.md`, `SummarizationMiddleware`) đã được chạy end-to-end với model **giả** (tool call ghi được `User.md`, đếm token đúng theo lượt), nhưng **chưa** chạy với provider thật vì môi trường không có API key.

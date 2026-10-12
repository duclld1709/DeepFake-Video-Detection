# Câu hỏi phỏng vấn: Deepfake Video Detection

Bộ 15 câu hỏi một nhà tuyển dụng (AI/ML Engineer, MLOps, Backend ML) có thể hỏi về dự án này,
kèm gợi ý trả lời bám theo code thật trong repo. Mỗi câu có ba phần: người phỏng vấn muốn kiểm
tra điều gì, gợi ý trả lời, và câu hỏi nối tiếp nên chuẩn bị.

> Mẹo chung: luôn trả lời theo khung **vấn đề → quyết định → bằng chứng (test, số liệu, file)**.
> Dự án này mạnh nhất ở phần "audit v1 → v2", vì nó cho thấy bạn biết tự tìm bug trong chính code
> của mình và không tin vào con số đẹp.

---

## Q1. Giới thiệu dự án trong 2 phút.

**Muốn kiểm tra:** khả năng tóm tắt, hiểu bài toán, biết dự án giải quyết gì cho ai.

**Gợi ý trả lời:**
- **Bài toán:** lừa đảo bằng video call deepfake (giả người thân, cán bộ). Người dùng cần một
  "ý kiến thứ hai" trong vài giây, và cần hệ thống dám nói "không chắc" khi bằng chứng yếu.
- **Mô hình:** ResNeXt-50 (ImageNet) trích đặc trưng từng frame mặt 112×112 → LSTM → lớp linear
  phân loại FAKE/REAL (`deepfake_detector/model.py`).
- **Pipeline:** chia video thành nhiều clip (`plan_clips`) → phát hiện và bám một khuôn mặt
  (`FaceTracker`) → crop, chuẩn hoá → model → điểm P(fake) cho từng frame, từng clip, cả video →
  verdict `fake / real / inconclusive` → Grad-CAM cho các frame đáng ngờ nhất.
- **Sản phẩm hoá:** thư viện Python, CLI, REST API FastAPI, UI Gradio, Docker, CI (ruff + pytest).
- **Điểm nhấn:** bản v1 là đồ án môn học. Khi viết lại thành v2, tôi tìm ra 9 bug, trong đó có bug
  LSTM khiến model **không hề mô hình hoá thời gian**. Mỗi bug được ghim bằng một test.

**Nối tiếp:** "Phần nào bạn tự làm, phần nào của nhóm?" Nên trả lời rõ ràng, trung thực.

---

## Q2. Bug LSTM thiếu `batch_first=True` là gì, và bạn xử lý thế nào mà không phải train lại?

**Muốn kiểm tra:** hiểu sâu shape tensor trong PyTorch, khả năng debug, tư duy tương thích ngược.

**Gợi ý trả lời:**
- `nn.LSTM` mặc định nhận `(seq, batch, feat)`. Code v1 đưa vào `(batch, seq, feat)` = `(1, 80, 2048)`.
  LSTM hiểu thành **chuỗi dài 1 với 80 mẫu độc lập**: mỗi frame đi qua đúng một bước LSTM từ
  trạng thái 0, rồi 80 output được lấy trung bình. Tức là model thực chất là CNN + MLP theo frame
  rồi average, không có thông tin thời gian.
- Lúc train với batch = 2, LSTM lại thấy chuỗi dài 2, nên hidden state của video thứ nhất **rò sang**
  video thứ hai ở cùng vị trí frame.
- Checkpoint đã được train trong chế độ "sai" đó, nên nếu sửa thẳng thì phân phối đầu vào của lớp
  linear đổi và model hỏng. Giải pháp là hai `temporal_mode` trong `DeepfakeDetector.temporal`:
  - `legacy`: reshape `(B*T, 1, D)`, mỗi frame một bước từ zero state. Tái tạo **chính xác** suy
    luận gốc, và hết rò rỉ giữa các video vì mỗi frame là một "batch item" riêng.
  - `sequence`: `batch_first=True`, hồi quy thật trên 80 frame. Dùng cho các lần train mới.
- Bằng chứng: `test_legacy_mode_reproduces_original_inference` và
  `test_original_leaks_across_batch_but_legacy_does_not` trong `tests/test_model.py`.

**Nối tiếp:** "Làm sao bạn phát hiện ra?" Đọc kỹ code khi viết test tương thích checkpoint, so output
model gốc với bản viết lại, rồi kiểm tra shape từng bước.

---

## Q3. Tại sao ở chế độ `legacy` bạn có điểm từng frame "chính xác", mà ở `sequence` thì không?

**Muốn kiểm tra:** nền tảng toán (tính tuyến tính), không nói quá về explainability.

**Gợi ý trả lời:**
- Ở `legacy`, mỗi frame có hidden state độc lập `h_t`. Logit của clip là `W · mean(h_t) + b`.
  Vì lớp linear là affine nên `W · mean(h_t) + b = mean(W·h_t + b)`: **logit clip đúng bằng trung
  bình logit từng frame**. Vì vậy `frame_logits` là một phân rã chính xác, không phải xấp xỉ
  (test `test_frame_logits_average_to_clip_logits`).
- Ở `sequence`, `h_t` phụ thuộc mọi frame trước nó, nên không tách được đóng góp riêng của từng
  frame. Code ném `RuntimeError` thay vì trả về con số gây hiểu lầm, schema để `fake_probability = None`.
- Lưu ý: logit phân rã chính xác, còn xác suất softmax thì không (mean of softmax ≠ softmax of mean).

---

## Q4. Bạn chọn frame nào để phân tích? Tại sao không lấy 80 frame đầu như v1?

**Muốn kiểm tra:** hiểu bài toán thực tế (deepfake cục bộ), trade-off độ phủ và chi phí.

**Gợi ý trả lời:**
- v1 chỉ xem khoảng 3 giây đầu, nên video bị chỉnh ở giữa hoặc cuối bị bỏ sót.
- `plan_clips` (`video.py`) chia `[0, frame_count)` thành `num_clips` đoạn bằng nhau (mặc định 3,
  API cho phép tối đa 10). Ở mỗi đoạn, analyzer đọc liên tiếp tới khi đủ 80 crop mặt hoặc hết đoạn.
  Đoạn ngắn hơn `sequence_length` thì không tạo, tránh clip vô nghĩa.
- Điểm video là **trung bình** các clip, đồng thời trả về **max** (`max_clip_fake_probability`) để
  bắt video chỉ bị giả một phần.
- Clip có ít hơn `min_face_ratio × 80` = 20 frame có mặt sẽ bị bỏ kèm warning. Không clip nào đủ
  thì ném `NoFaceDetectedError` → HTTP 422.

**Nối tiếp:** "Sao không lấy max làm verdict?" Max nhạy hơn nhưng tăng false positive, vì một clip
nhiễu (mờ, nén nặng) là đủ kéo cả video thành fake. Tôi chọn mean làm verdict và báo max để người
dùng tự xem. Muốn quyết định dựa trên dữ liệu thì cần chạy benchmark để chọn.

---

## Q5. Face tracker hoạt động thế nào? Vì sao không dùng `faces[0]`?

**Muốn kiểm tra:** chú ý tới chi tiết pipeline ngoài model, hiểu IoU.

**Gợi ý trả lời:**
- Thứ tự `faces` từ detector là tuỳ ý. Video có nhiều người thì crop nhảy qua lại giữa các khuôn
  mặt, chuỗi 80 frame trở thành vô nghĩa với LSTM.
- `FaceTracker.update` (`faces.py`): frame đầu chọn **mặt lớn nhất**. Các frame sau chọn box có
  **IoU cao nhất** với box trước; nếu IoU < 0.1 (mất dấu) thì quay lại chọn mặt lớn nhất.
- Tối ưu tốc độ: detection chạy trên frame thu nhỏ còn cạnh dài 640 px (`_Downscaling`), rồi map
  box về toạ độ gốc. `detect_every = N` cho phép tái dùng box giữa các lần detect.
- Backend: `face_recognition` (dlib HOG, đúng detector lúc train), fallback OpenCV Haar nếu thiếu
  dlib. Khi fallback, kết quả kèm warning vì crop khác phân phối lúc train.

**Nối tiếp:** "Bottleneck ở đâu?" Theo `timings` mẫu trong README, `faces_ms` khoảng 3.8 s lớn hơn
`inference_ms` khoảng 2.9 s. Hướng tối ưu: `detect_every > 1`, detector nhanh hơn (RetinaFace, YuNet),
batch hoá detection.

---

## Q6. Vì sao bạn giữ thứ tự kênh màu BGR? Trông như một bug.

**Muốn kiểm tra:** hiểu train/serve skew, không "sửa" theo phản xạ.

**Gợi ý trả lời:**
- Lúc train, frame đọc bằng OpenCV (BGR) rồi đưa thẳng vào `ToPILImage` không đổi màu. Checkpoint
  vì vậy đã học trên ảnh BGR chuẩn hoá bằng mean/std ImageNet.
- Đổi sang RGB trông "đúng" nhưng làm lệch phân phối đầu vào mà trọng số đã học, nên độ chính xác có
  thể giảm âm thầm. Tôi ghi rõ lý do trong docstring `preprocessing.py`, README và model card.
- Cùng nguyên tắc, `training/extract_faces.py` dùng **chính detector và tracker của inference** để
  crop dữ liệu train, và `VideoClipDataset` dùng chung `build_transform`, để train và serve thấy cùng
  một kiểu đầu vào.

**Nối tiếp:** "Khi train lại từ đầu thì sao?" Có thể chuyển sang RGB cho chuẩn, nhưng phải đổi đồng
thời ở cả train lẫn inference và ghi vào spec của checkpoint.

---

## Q7. Grad-CAM được cài đặt thế nào? Làm sao giữ bộ nhớ thấp với clip 80 frame?

**Muốn kiểm tra:** hiểu autograd, explainability, tối ưu bộ nhớ.

**Gợi ý trả lời (`explain.py`):**
- Mục tiêu là **log-odds fake vs real**: `logit[FAKE] - logit[REAL]`. Ở `legacy` tính theo từng
  frame, ở `sequence` theo cả clip.
- Chỉ **k frame được chọn** đi lại qua CNN với gradient. Các frame còn lại dùng `pooled` đã tính
  sẵn, ở dạng hằng số (`detach`). Vì đặc trưng của frame khác không phụ thuộc pixel frame được chọn,
  gradient của frame được chọn vẫn **chính xác**, còn bộ nhớ tỉ lệ với k thay vì 80.
- Grad-CAM chuẩn: trọng số kênh = trung bình gradient theo không gian, CAM = ReLU(Σ w·A), nội suy
  bilinear về 112×112, chuẩn hoá min-max từng frame, rồi phủ colormap JET lên ảnh mặt.
- Hai chi tiết kỹ thuật:
  - `_classify` dùng `torch.no_grad()` chứ không dùng `inference_mode()`, vì tensor `pooled` được
    tái sử dụng trong Grad-CAM, mà inference tensor không dùng được trong autograd.
  - Trên GPU phải tắt cuDNN khi backward, vì cuDNN không cho backward RNN ở chế độ `eval`.
- Frame được chọn: theo điểm từng frame cao nhất (legacy), hoặc rải đều trong clip đáng ngờ nhất
  (sequence).

**Nối tiếp:** "Grad-CAM có đáng tin không?" Nó chỉ cho thấy vùng ảnh hưởng tới quyết định, không
chứng minh vùng đó bị chỉnh sửa. Đây là hỗ trợ trực quan, không phải bằng chứng pháp y.

---

## Q8. "Inconclusive" được quyết định thế nào? Có đáng tin không?

**Muốn kiểm tra:** tư duy sản phẩm có trách nhiệm, hiểu calibration.

**Gợi ý trả lời:**
- `_verdict` (`inference.py`): nếu `|P(fake) − threshold| < inconclusive_margin` (mặc định 0.5 và
  0.1, tức P trong khoảng (0.4, 0.6)) thì trả `inconclusive`. Ngoài khoảng đó, `>= threshold` là fake.
- Lý do: với lừa đảo, sai "real" là nguy hiểm nhất vì nó cho kẻ gian thêm uy tín. Hệ thống nên thừa
  nhận không chắc thay vì đoán.
- Trung thực: xác suất **chưa được calibrate**, nên dải ±0.1 hiện là heuristic. Code đã có
  `expected_calibration_error` trong `metrics.py`, và roadmap là temperature scaling trên tập
  held-out để dải này có ý nghĩa thống kê.
- Kết quả còn kèm `face_coverage` và `warnings` (sai detector, clip thiếu frame) để người dùng biết
  độ tin cậy của đầu vào.

**Nối tiếp:** "Temperature scaling làm thế nào?" Học một scalar T trên tập validation bằng cách tối
thiểu NLL của `softmax(logits / T)`. T không đổi thứ hạng nên AUC giữ nguyên, chỉ ECE thay đổi.

---

## Q9. Con số "90% accuracy" của v1 có đáng tin không?

**Muốn kiểm tra:** hiểu data leakage, trung thực với kết quả.

**Gợi ý trả lời:**
- Không nên đọc nó như accuracy thực tế. Có ba lý do:
  1. Đó là **validation accuracy trên split ngẫu nhiên theo video**. Trong dataset deepfake, nhiều
     video fake được sinh từ cùng một video gốc hoặc cùng một người, nên split ngẫu nhiên để mô hình
     "đã thấy" khuôn mặt đó ở tập train (leakage).
  2. Đo bằng code có bug LSTM và bug AUC (v1 truyền **nhãn cứng** vào `roc_auc_score`).
  3. Không có test cross-dataset, mà đó là nơi detector deepfake thường sụp đổ.
- v2 làm gì: `split_samples` (`training/data.py`) dùng `GroupShuffleSplit` theo cột `group` (video
  gốc hoặc danh tính), test `test_group_split_has_no_leakage`; thêm test set held-out; AUC/EER tính
  từ xác suất; model selection theo val AUC.
- Tôi ghi rõ trong README là số cross-dataset **chưa được công bố** vì benchmark chưa chạy. Tôi không
  muốn đưa ra con số mà mình không bảo vệ được.

---

## Q10. Quy trình train v2 có gì khác v1?

**Muốn kiểm tra:** kinh nghiệm train thực tế, các kỹ thuật fine-tune.

**Gợi ý trả lời (`training/train.py`):**
- **Fine-tune có kiểm soát:** AdamW với **learning rate phân tầng** (backbone = 0.1 × head), **đóng
  băng backbone** ở epoch đầu để head ổn định trước, cosine annealing.
- **Ổn định:** AMP + `GradScaler` (CUDA), gradient clipping norm 1.0.
- **Mất cân bằng lớp:** `CrossEntropyLoss` có trọng số nghịch đảo tần suất, vì dataset deepfake
  thường nhiều fake hơn.
- **Chọn model:** theo **val AUC** (không phụ thuộc ngưỡng), early stopping patience 5, nạp lại
  `best.pt` rồi chấm trên test set.
- **Tái lập:** seed mọi RNG, lưu danh sách file từng split, `history.json`, `summary.json`.
- **Bỏ cái hỏng của v1:** không còn `torch.cuda.LongTensor` cứng (chạy được trên CPU), không còn
  `input()` chặn automation. `--init-from` fine-tune từ checkpoint cũ.

---

## Q11. Augmentation của bạn có gì đặc biệt cho bài toán video?

**Muốn kiểm tra:** hiểu domain, không áp augmentation theo thói quen.

**Gợi ý trả lời (`ClipAugment`):**
- **Cùng một phép biến đổi cho mọi frame trong clip** (flip, độ sáng, blur, JPEG quality được bốc
  một lần cho cả clip). Nếu random từng frame thì sẽ tạo ra **nhấp nháy giữa các frame**, mà bất
  nhất thời gian lại chính là tín hiệu deepfake. Model sẽ học nhầm.
- **Nén lại JPEG (quality 35–90) và blur** quan trọng nhất: video lừa đảo thực tế đi qua Zalo,
  Messenger, TikTok và bị nén lại, làm mất các artefact tần số cao mà detector train trên file sạch
  hay dựa vào.
- `random_start=True` cho tập train để thấy nhiều đoạn khác nhau; video ngắn được pad bằng frame
  cuối thay vì crash.

---

## Q12. Bạn đánh giá độ bền (robustness) của model thế nào?

**Muốn kiểm tra:** thiết kế evaluation hướng sản phẩm.

**Gợi ý trả lời (`training/evaluate.py`):**
- Benchmark chạy **đúng pipeline sản phẩm** (detect, track, model, verdict) trên video thô, không
  chỉ chấm classifier trên crop đã chuẩn bị sẵn. Con số vì vậy mô tả sản phẩm.
- Mỗi video được biến đổi bằng ffmpeg: H.264 CRF 23/28/35, thu nhỏ 0.5×, 15 FPS.
- Mỗi perturbation báo: AUC, EER, accuracy, F1, ECE, **coverage** (tỉ lệ video tìm được mặt),
  **tỉ lệ inconclusive**, latency p50. Video lỗi được ghi `status` để không làm sai metric.
- Khuyến nghị **cross-dataset**: train FF++, test Celeb-DF hoặc DFDC.
- EER (điểm FPR = FNR) và ECE được cài trong `metrics.py`.

**Nối tiếp:** "Tại sao AUC chứ không phải accuracy?" AUC không phụ thuộc ngưỡng và ít bị ảnh hưởng
bởi mất cân bằng lớp. Accuracy phụ thuộc ngưỡng 0.5, vốn chưa được calibrate.

---

## Q13. Thiết kế REST API: làm sao chịu được upload lớn và tải đồng thời?

**Muốn kiểm tra:** kỹ năng backend khi serve model nặng.

**Gợi ý trả lời (`deepfake_detector/api.py`):**
- **Nạp model một lần** trong `lifespan`. Nếu lỗi, server **vẫn chạy** và `/health` trả 503 kèm lý
  do, thay vì crash-loop không rõ nguyên nhân.
- **Upload:** kiểm tra đuôi file (415), ghi xuống file tạm theo **chunk 1 MB** và dừng ngay khi vượt
  `max_upload_mb` (413), không đọc cả file vào RAM. File rỗng trả 400.
- **Đồng thời:** endpoint là hàm `def` đồng bộ nên FastAPI chạy nó trong threadpool, không chặn event
  loop. `BoundedSemaphore(max_concurrency)` giới hạn số phân tích chạy song song (mặc định 1 trên
  CPU); đợi quá 120 s thì trả 503 "server busy".
- **Lỗi có kiểu:** `NoFaceDetectedError` → 422, `VideoTooLongError` → 413, `VideoReadError` → 400.
- Middleware `X-Process-Time-Ms`. `app` được tạo lười qua module `__getattr__`, nên import module
  trong test không tự nạp model.

**Nối tiếp:** "Video dài 10 phút thì sao?" Request đồng bộ dễ timeout. Roadmap là job queue bất đồng
bộ (Celery hoặc RQ: trả `job_id`, client poll), ngoài ra có thể export ONNX hoặc quantize để tăng
tốc CPU.

---

## Q14. Bạn test một hệ thống deep learning thế nào khi CI không có GPU và không tải được weights?

**Muốn kiểm tra:** kỹ năng software engineering cho ML.

**Gợi ý trả lời (`tests/`):**
- **Video tổng hợp:** `conftest.write_video` tạo video nhỏ có một hình chữ nhật di chuyển làm "mặt".
- **Detector giả:** `StubDetector` trả box cố định, hoặc chỉ trả ở một số frame để test
  `face_coverage` và trường hợp không có mặt.
- **Model trọng số ngẫu nhiên** (`pretrained_backbone=False`) để test shape, tính tất định, tính
  nhất quán (điểm frame trung bình ra điểm clip), PNG Grad-CAM hợp lệ.
- **Test hồi quy cho từng bug của v1:** tái tạo module gốc và so với `legacy` mode.
- **API:** `TestClient` với factory inject analyzer, kiểm tra 415/413/422/503.
- **Train smoke test:** chạy `training.train.main` vài bước trên dữ liệu giả.
- CI: GitHub Actions cài torch bản CPU, ffmpeg, chạy `ruff check`, `ruff format --check`, `pytest`
  (khoảng 25 s).

---

## Q15. Hạn chế lớn nhất của dự án là gì? Nếu có thêm thời gian bạn làm gì?

**Muốn kiểm tra:** tự nhận thức, tư duy định hướng.

**Gợi ý trả lời (thẳng thắn):**
1. **Chưa có số cross-dataset.** Việc quan trọng nhất là chạy benchmark FF++ → Celeb-DF/DFDC rồi
   công bố.
2. **Checkpoint là `legacy`,** thực chất là CNN theo frame. Cần train lại ở `sequence` mode với
   group split và augmentation nén, rồi so với legacy.
3. **Backbone:** thử đặc trưng CLIP hoặc DINOv2 đóng băng, vì các nghiên cứu gần đây cho thấy chúng
   tổng quát hoá tốt hơn trên phương pháp sinh chưa thấy. So EfficientNet về accuracy và latency.
4. **Calibration:** temperature scaling để dải inconclusive có ý nghĩa.
5. **Đa phương thức:** deepfake hiện kèm giọng giả. Nên thêm nhánh audio và kiểm tra khớp khẩu hình
   (lip-sync).
6. **Fairness:** chưa đo hiệu năng theo màu da, tuổi, ánh sáng (đã ghi trong model card).

---

## Câu hỏi nhanh nên thuộc

| Câu hỏi | Trả lời ngắn |
|---|---|
| Input của model? | 80 crop mặt liên tiếp, 112×112, BGR, chuẩn hoá ImageNet |
| Class index? | 0 = FAKE, 1 = REAL (`config.FAKE_INDEX`) |
| Bug `bias` của LSTM? | v1 truyền `bidirectional` vào vị trí thứ 4, tức `bias`, nên LSTM không có bias. v2 dùng keyword argument và tự phát hiện `lstm_bias` từ state dict |
| Dropout v1 sai ở đâu? | Đặt trên logits; v2 đặt trên feature trước classifier |
| Format checkpoint? | `deepfake-detector/v2`: spec + state_dict + metadata; vẫn đọc được state dict thô của v1 |
| Cấu hình runtime? | Dataclass `DetectorConfig`, ghi đè bằng biến môi trường `DEEPFAKE_*` |

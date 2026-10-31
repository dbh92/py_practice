# Luồng làm việc và vận hành hệ thống kiểm tra OK/NG

Tài liệu này mô tả **bức tranh tổng thể** của dự án:

- dữ liệu đi qua những bước nào,
- mỗi thành phần làm gì và chạy ở đâu,
- các bước được kiểm chứng với nhau ra sao,
- cách vận hành hằng ngày trên chuyền và cách cải tiến mô hình theo thời gian.

Câu lệnh chi tiết của từng bước nằm trong [README.md](README.md). Tài liệu này chỉ nhắc lại lệnh khi cần thiết để hiểu luồng.

---

## 1. Tổng quan

Bài toán: nhìn ảnh chụp sản phẩm trên dây chuyền rồi kết luận **OK** (đạt) hay **NG** (lỗi), thay cho hoặc hỗ trợ người kiểm tra bằng mắt.

Dự án chia thành **hai giai đoạn**, chạy ở hai nơi khác nhau:

```
 GIAI ĐOẠN 1: PHÁT TRIỂN (máy văn phòng, có PyTorch)       GIAI ĐOẠN 2: VẬN HÀNH (máy chuyền, KHÔNG cần PyTorch)
 ─────────────────────────────────────────────────       ─────────────────────────────────────────────────────

  Ảnh thu thập                                              Camera
      │                                                       │ lưu ảnh vào thư mục
      ▼                                                       ▼
  data/raw/OK, NG                                          InspectionApp (WPF, C#)
      │ split_dataset.py                                      │ HTTP POST /predict (gửi ảnh)
      ▼                                                       ▼
  data/split/train, val, test                              Service FastAPI (service/app.py)
      │ train.py                                              │ predict_onnx.py + model.onnx
      ▼                                                       ▼
  outputs/best.pt  ── evaluate.py / predict.py             Kết quả JSON: OK / NG
      │ export_onnx/export.py                                 │
      ▼                                                       ▼
  model.onnx + model.json  ═══════ chép sang ═══════►      Hiển thị + âm báo + log CSV
                                                              │
                     ◄═══════ ảnh NG / HOLD / báo nhầm ═══════┘
                              (quay lại để train lại)
```

Giai đoạn 1 tạo ra **một sản phẩm duy nhất** để mang đi: cặp file `model.onnx` và `model.json`. Giai đoạn 2 chỉ dùng cặp file này, và không cần biết mô hình được train như thế nào.

---

## 2. Các thành phần

| Thành phần | Vai trò | Chạy ở đâu | Cần cài | Đầu vào → Đầu ra |
|---|---|---|---|---|
| `config.py` | **Nơi duy nhất** chứa thiết lập | Máy văn phòng | — | — |
| `split_dataset.py` | Chia ảnh 70/15/15 theo từng lớp | Máy văn phòng | PyTorch env | `data/raw/` → `data/split/` |
| `dataset.py`, `model.py` | Đọc và biến đổi ảnh; dựng mô hình ResNet18 | Máy văn phòng | PyTorch env | (thư viện dùng chung) |
| `train.py` | Huấn luyện, chọn epoch tốt nhất | Máy văn phòng | PyTorch env | `data/split/` → `outputs/best.pt` |
| `evaluate.py` | Đánh giá trên tập test hoặc val; chứa **quy tắc phán định chung** | Máy văn phòng | PyTorch env | `best.pt` → ma trận nhầm lẫn, danh sách ảnh sai |
| `predict.py` | Dự đoán ảnh bất kỳ, xem kết quả bằng hình | Máy văn phòng | PyTorch env | ảnh → OK/NG |
| `export_onnx/export.py` | Xuất ONNX và **tự kiểm chứng** khớp PyTorch | Máy văn phòng | PyTorch env + onnx | `best.pt` → `model.onnx`, `model.json` |
| `export_onnx/predict_onnx.py` | Bộ phán định ONNX độc lập (xử lý ảnh + suy luận + quyết định) | Cả hai nơi | onnxruntime, numpy, pillow | ảnh → OK/NG |
| `service/app.py` | Web service nhận ảnh, trả kết quả, ghi log | Máy service (chuyền) | `service/requirements.txt` | ảnh (HTTP) → JSON + log CSV |
| `service/test_client.py` | Gửi thử ảnh; so sánh kết quả service với PyTorch | Máy văn phòng | requests (+ PyTorch nếu `--compare`) | ảnh → báo cáo khớp / không khớp |
| `inspection_app/` | Giao diện cho người kiểm tra | Máy trạm (chuyền) | .NET 8 Desktop Runtime | ảnh từ camera → hiển thị OK/NG/HOLD |

**Điểm mấu chốt về thiết kế:**

- **Chỉ một quy tắc phán định** (mục 3.4). `train.py`, `predict.py` và `evaluate.py` đều gọi chung hàm trong `evaluate.py`. `predict_onnx.py` chép lại đúng quy tắc đó, và `export.py` kiểm chứng rằng hai bên khớp nhau.
- **Chỉ một bộ xử lý ảnh cho vận hành**: service import thẳng `OnnxClassifier` từ `export_onnx/predict_onnx.py`, không viết lại. Nhờ vậy, những gì đã kiểm chứng ở bước export chính là những gì chạy trên chuyền.
- **Mọi thông số vận hành nằm trong `model.json`**: danh sách lớp, kích thước ảnh, mean/std, ngưỡng NG. Service và predict_onnx đều không đọc `config.py`, nên không thể bị lệch so với mô hình đang dùng.

---

## 3. Luồng phát triển mô hình (giai đoạn 1)

### 3.1 Sơ đồ

```
 [1] Thu thập ảnh ──► [2] Chia dữ liệu ──► [3] Huấn luyện ──► [4] Đánh giá ──► [5] Chọn ngưỡng NG
                                                ▲                  │                   │
                                                │   chưa đạt       │                   ▼
                                                └──────────────────┘          [6] Xuất ONNX + kiểm chứng
                                                (thêm ảnh, sửa nhãn,                    │
                                                 chỉnh augmentation)                   ▼
                                                                          [7] Thử qua service (--compare)
                                                                                        │
                                                                                        ▼
                                                                               Mang sang máy chuyền
```

### 3.2 Từng bước: làm gì, ra gì, khi nào đạt

| Bước | Lệnh | Đầu ra | Tiêu chí đạt / điều cần xem |
|---|---|---|---|
| 1. Thu thập | Bỏ ảnh vào `data/raw/<lớp>/` | Ảnh gốc, tên thư mục = tên lớp | Chụp giống điều kiện thật; NG đủ đa dạng; nhãn đúng |
| 2. Chia | `split_dataset.py [--overwrite]` | `data/split/{train,val,test}/<lớp>/` | Lớp nào cũng có ảnh ở cả 3 tập (`dataset.py` in số lượng) |
| 3. Huấn luyện | `train.py [--resume]` | `outputs/best.pt`, `last.pt`, `history.csv`, `training_curves.png` | val F1 tăng rồi ổn định; val loss không tăng ngược (overfit) |
| 4. Đánh giá | `evaluate.py` | `confusion_matrix_test.png`, `misclassified_test.csv` | **NG bị lọt (escape)** chấp nhận được; xem lại từng ảnh sai |
| 5. Chọn ngưỡng | `evaluate.py --split val --ng-threshold 0.3` | Số liệu của từng ngưỡng | Cân bằng giữa NG lọt và OK báo nhầm theo yêu cầu QC |
| 6. Xuất ONNX | `export_onnx/export.py` | `model.onnx`, `model.json` | In `Verification passed` (xác suất lệch < 0,0001, kết luận trùng 100%) |
| 7. Thử qua service | `service/test_client.py ... --compare` | Báo cáo so sánh | In `Compare passed` |

### 3.3 Các lựa chọn khi huấn luyện, và lý do

| Lựa chọn | Lý do |
|---|---|
| **ResNet18 pretrained** (transfer learning) | Đã học đặc trưng ảnh từ ImageNet, nên vài trăm ảnh mỗi lớp vẫn đủ tốt |
| **Chia stratified, `SEED = 42`** | Mỗi tập giữ đúng tỉ lệ OK/NG; chạy lại lần nào cũng ra cùng kết quả chia |
| **Trọng số lớp** (`USE_CLASS_WEIGHTS`) | NG thường ít hơn OK. Không có trọng số, mô hình dễ "lười", đoán OK hết cho an toàn |
| **Chọn mô hình theo macro F1 trên val** (không theo accuracy) | Khi 90% là OK, đoán OK hết vẫn đạt accuracy 90%. Macro F1 buộc mô hình phải giỏi cả hai lớp |
| **Augmentation chỉ trên train** (lật, xoay ±10°, sáng/tương phản ±20%) | Mô phỏng dao động thật trên chuyền. Val và test giữ nguyên để đánh giá trung thực |
| **Early stopping** (patience 10) | Dừng khi val F1 không còn tiến bộ, tiết kiệm thời gian train trên CPU |

### 3.4 Quy tắc phán định (dùng chung ở mọi nơi)

```
Mô hình ──► xác suất từng lớp, vd {NG: 0.27, OK: 0.73}

Nếu NG_THRESHOLD = None:   kết luận = lớp có xác suất cao nhất
Nếu NG_THRESHOLD = t:      P(NG) >= t  →  NG
                           P(NG) <  t  →  lớp có xác suất cao nhất trong các lớp còn lại
```

Hạ ngưỡng thì **bắt được nhiều NG hơn** nhưng **báo nhầm OK nhiều hơn**. Ở QC, NG bị lọt ra ngoài thường tốn kém hơn một lần báo nhầm, nên ngưỡng nên chọn theo chi phí thực tế của hai loại sai này. Ngưỡng chọn ở bước 5 được ghi vào `model.json` lúc export, và service dùng đúng ngưỡng đó.

### 3.5 Ba tập dữ liệu và vai trò

| Tập | Vai trò | Có ảnh hưởng tới mô hình không |
|---|---|---|
| train (70%) | Mô hình học từ đây | Có, trực tiếp |
| val (15%) | Chọn epoch tốt nhất, chọn ngưỡng NG | Có, gián tiếp (qua các lựa chọn) |
| test (15%) | Đánh giá sau cùng, như ảnh "chưa từng thấy" | **Không**, nếu giữ nguyên tắc chỉ xem ở cuối |

Khi đã đánh giá xong và quyết định **gộp test vào train/val để train lại** (như đã làm với mô hình hiện tại), con số đánh giá cũ không còn đúng cho mô hình mới. Lúc đó, độ tin cậy của mô hình mới phải đo bằng ảnh thật từ chuyền (mục 7).

---

## 4. Luồng phán định trên chuyền (giai đoạn 2)

### 4.1 Đường đi của một ảnh

```
 Camera ghi file D:\Camera\abc.jpg
   │
   ▼
 InspectionApp: FileSystemWatcher thấy file mới (sự kiện Created hoặc Renamed)
   │  chuyển về luồng giao diện
   │  chờ tới khi mở được file độc quyền (camera đã ghi xong, chờ tối đa 2 giây)
   │  xếp hàng: mỗi lúc chỉ kiểm tra 1 ảnh
   ▼
 Đọc nguyên byte ảnh gốc ──► POST /predict (multipart: file, station_id, product_id)
   │                         mở kết nối mới, xong là đóng; chờ tối đa 5 giây
   ▼
 Service FastAPI
   │  kiểm tra kích thước (> 20 MB → lỗi 413), mở ảnh (hỏng → lỗi 400)
   │  OnnxClassifier: RGB → resize 224 bilinear → /255 → chuẩn hóa mean/std → ONNX → softmax
   │  decide: áp quy tắc phán định (mục 3.4) với ngưỡng trong model.json
   │  ghi 1 dòng log CSV (+ lưu ảnh nếu bật --save-images)
   ▼
 JSON: result, probabilities, p_ng, request_id, model_version, inference_ms
   │
   ▼
 InspectionApp: ô kết quả OK (xanh) / NG (đỏ) / HOLD (cam), âm báo khi NG/HOLD,
                cập nhật thống kê, thêm vào lịch sử, đèn trạng thái kết nối
```

Ngoài theo dõi thư mục, người dùng có thể chọn ảnh, kiểm tra cả thư mục, hoặc kéo thả ảnh. Từ bước "Đọc nguyên byte ảnh" trở đi, mọi cách đều đi chung một đường.

### 4.2 Thời gian (đo trên máy này, CPU, ResNet18 ở 224 pixel)

| Đoạn | Thời gian |
|---|---|
| Suy luận ONNX (chỉ mô hình) | khoảng 7 ms |
| Trong service (mở ảnh + xử lý + suy luận) | khoảng 10–13 ms |
| Khứ hồi từ InspectionApp (gồm mở kết nối mới) | khoảng 20–55 ms |
| Service đã tắt, trước khi báo HOLD | khoảng 2 giây (Windows thử kết nối lại) |
| Service treo hoặc quá tải, trước khi báo HOLD | 5 giây (timeout) |

### 4.3 Nguyên tắc an toàn: không chắc thì HOLD, không bao giờ tự cho OK

Mỗi lần kiểm tra **bắt đầu với kết quả HOLD**. Kết quả chỉ đổi thành OK hoặc NG khi service trả lời thành công.

| Tình huống | Kết quả | Ghi chú trên giao diện |
|---|---|---|
| Service trả lời bình thường | OK / NG | — |
| Service chưa chạy, tắt, sai địa chỉ, mất mạng | **HOLD** | "Không kết nối được service: ..." + đèn đỏ |
| Service quá 5 giây không trả lời | **HOLD** | "Quá thời gian chờ service (timeout)" + đèn đỏ |
| Ảnh hỏng, không phải ảnh | **HOLD** | "Service báo lỗi 400: Cannot read image: ..." |
| Ảnh lớn hơn 20 MB | **HOLD** | "Service báo lỗi 413: ..." |
| Không đọc được file (bị khóa, bị xóa) | **HOLD** | "Không đọc được file ảnh: ..." |
| Lỗi bất ngờ khác | **HOLD** | "Lỗi: ..." |

**Hàng HOLD phải giữ lại cho người kiểm tra xem bằng mắt, không được cho qua.**

### 4.4 Vì sao mỗi lần gửi ảnh mở một kết nối mới

Uvicorn tự đóng kết nối nào rảnh quá 5 giây. Nếu InspectionApp giữ kết nối cũ để dùng lại, thì khi chuyền dừng khoảng 5 giây, ảnh tiếp theo có thể bị gửi đúng vào kết nối vừa bị đóng. Khi đó chương trình báo **HOLD giả**: đo được 5/40 lần lỗi. Mở kết nối mới cho mỗi ảnh chỉ tốn thêm vài mili giây và cho 0/40 lần lỗi.

Cũng vì vậy, InspectionApp **không gọi service định kỳ**. Nó chỉ gọi service ở 3 lúc:

- khi mở chương trình,
- khi bấm **Kết nối**,
- khi có ảnh cần kiểm tra.

Đèn trạng thái phản ánh **lần gọi gần nhất**.

---

## 5. Chuỗi kiểm chứng: vì sao tin được kết quả trên chuyền

Mỗi lần chuyển sang một môi trường mới đều có một bước so sánh với môi trường trước đó:

```
 PyTorch (best.pt)  ══ export.py: Verification ══►  ONNX (predict_onnx.py)  ══ test_client --compare ══►  Service HTTP
    predict.py            lệch < 0,0001                                         cùng kết luận                 │
    evaluate.py           kết luận trùng 100%                                   (dùng ngưỡng của service)    ▼
                                                                                                    InspectionApp hiển thị
                                                                                                    đúng JSON service trả về
```

| Kiểm chứng | Phát hiện được gì | Kết quả gần nhất |
|---|---|---|
| `export.py` (tự động mỗi lần xuất) | Xuất ONNX lỗi, xử lý ảnh bên ONNX khác bên PyTorch | 202/202 ảnh khớp |
| `test_client.py --compare` | Service đang chạy mô hình cũ, quên khởi động lại, lỗi truyền ảnh qua HTTP | 203/203 khớp, lệch tối đa 7e-06 |
| Chạy thử InspectionApp với ảnh đã biết đáp án | Giao diện hiển thị sai, theo dõi thư mục không bắt được ảnh | OK, NG, HOLD đều hiện đúng |

**Những thứ phải giữ giống nhau** giữa huấn luyện và vận hành: chuyển ảnh sang RGB, resize bilinear, chia 255, mean/std của ImageNet, thứ tự lớp, quy tắc phán định. Nếu sửa một bên thì phải sửa cả bên kia, rồi chạy lại các bước kiểm chứng.

---

## 6. Triển khai lên chuyền

### 6.1 Mô hình máy

```
 ┌──────────────────────── Máy service (1 máy) ────────────────────────┐
 │ export_onnx/predict_onnx.py, model.onnx, model.json                   │
 │ service/app.py, requirements.txt                                      │
 │ python service/app.py --host 0.0.0.0 --port 8000 [--save-images]      │
 │ service/logs/predictions_<ngày>.csv                                   │
 └───────────────────────────────▲──────────────────────────────────────┘
                                 │ LAN, cổng 8000
         ┌───────────────────────┼───────────────────────┐
 ┌───────┴──────┐        ┌───────┴──────┐        ┌───────┴──────┐
 │ Máy trạm ST01│        │ Máy trạm ST02│        │      ...      │
 │ InspectionApp│        │ InspectionApp│        │               │
 │ --watch Cam  │        │ --watch Cam  │        │               │
 └──────────────┘        └──────────────┘        └───────────────┘
```

Service và InspectionApp có thể chạy chung một máy. Khi đó dùng địa chỉ mặc định `http://127.0.0.1:8000`, và service không cần `--host 0.0.0.0`.

### 6.2 Chép gì, cài gì

| Máy | Chép | Cài |
|---|---|---|
| Máy service | `export_onnx/` (3 file: `predict_onnx.py`, `model.onnx`, `model.json`) và `service/` (`app.py`, `requirements.txt`), **giữ nguyên cấu trúc 2 thư mục cạnh nhau** | Python 3.10, `pip install -r service/requirements.txt` |
| Máy trạm | Cả thư mục `inspection_app\bin\Release\net8.0-windows\` | .NET 8 Desktop Runtime |

Không cần PyTorch, `config.py`, hay thư mục `data/` trên máy chuyền.

### 6.3 Thứ tự khởi động

1. Bật service trên máy service. Đợi đến khi in ra dòng báo uvicorn đang chạy (mô hình được nạp và chạy thử ngay lúc khởi động).
2. Mở InspectionApp trên máy trạm bằng shortcut có sẵn tham số, ví dụ:
   `InspectionApp.exe --url http://192.168.1.10:8000 --station ST01 --watch D:\Camera`
3. Kiểm tra đèn xanh và dòng thông tin mô hình (tên, thời điểm xuất, epoch, ngưỡng NG) đúng với bản mô hình đã được duyệt.

### 6.4 Checklist trước khi đưa một mô hình vào chuyền

- [ ] `evaluate.py` cho số NG lọt và số báo nhầm nằm trong mức QC chấp nhận
- [ ] Đã xem lại từng ảnh trong `misclassified_*.csv`
- [ ] Ngưỡng NG đã chọn trên tập val và được ghi vào `model.json` (export với `--ng-threshold` hoặc đặt `NG_THRESHOLD` trong `config.py`)
- [ ] `export.py` in `Verification passed`
- [ ] `test_client.py --compare` in `Compare passed` trên đúng máy service
- [ ] Đã giữ lại bản `model.onnx` và `model.json` cũ để có thể quay lại (rollback)
- [ ] Chạy thử InspectionApp với vài ảnh OK/NG đã biết đáp án, và thử tắt service để thấy HOLD
- [ ] Nếu máy trạm khác máy service: service chạy với `--host 0.0.0.0`, và tường lửa Windows mở cổng 8000
- [ ] Phần mềm camera ghi file vào đúng thư mục đang theo dõi. Tốt nhất là ghi ra tên tạm rồi đổi tên, vì InspectionApp có bắt cả sự kiện đổi tên

---

## 7. Vận hành hằng ngày

### 7.1 Theo vai trò

| Vai trò | Việc cần làm |
|---|---|
| **Người kiểm tra (ca)** | Nhìn ô kết quả: OK thì cho qua; NG thì loại; **HOLD thì giữ lại và kiểm tra bằng mắt**. Nhập Mã SP / Lô khi đổi lô. Báo kỹ sư khi HOLD xuất hiện liên tục hoặc khi đèn đỏ |
| **Kỹ sư QC / người phụ trách mô hình** | Đầu ca: bật service, kiểm tra kết nối. Cuối ngày hoặc cuối tuần: xem log, xuất CSV lịch sử, gom ảnh nghi ngờ. Định kỳ: đánh giá lại và train lại mô hình (mục 8) |

### 7.2 Đầu ca, trong ca, cuối ca

**Đầu ca:**

- Bật service, sau đó mở InspectionApp.
- Kiểm tra đèn xanh, tên mô hình và mã trạm.
- Thử 1 ảnh OK và 1 ảnh NG mẫu đã biết đáp án.

**Trong ca:**

- Theo dõi tỉ lệ NG và số HOLD trên khung thống kê.
- Nếu tỉ lệ NG tăng đột ngột, kiểm tra camera và ánh sáng **trước**, rồi mới nghi ngờ sản phẩm. Mô hình rất nhạy với thay đổi điều kiện chụp.

**Cuối ca:**

- Bấm **Xuất CSV...** để lưu lịch sử của ca.
- Log của service đã tự ghi theo ngày trong `service/logs/predictions_<ngày>.csv`.

### 7.3 Xử lý sự cố

| Hiện tượng | Nguyên nhân thường gặp | Xử lý |
|---|---|---|
| Mọi ảnh đều HOLD, đèn đỏ, khứ hồi khoảng 2000 ms | Service tắt hoặc sai địa chỉ | Bật lại service, kiểm tra URL, bấm **Kết nối** |
| Mọi ảnh đều HOLD, khứ hồi khoảng 5000 ms | Service treo hoặc máy service quá tải | Khởi động lại service, kiểm tra CPU của máy service |
| HOLD lẻ tẻ, ghi chú "Service báo lỗi 400" | File ảnh hỏng hoặc camera ghi chưa xong | Kiểm tra phần mềm camera (nên ghi tên tạm rồi đổi tên) |
| Ảnh mới không được kiểm tra | Chưa bật theo dõi, hoặc sai thư mục, hoặc camera ghi vào thư mục con | Xem dòng "Đang theo dõi: ..." trên giao diện |
| Service không khởi động: `[Errno 10048]` | Cổng 8000 đang bị service cũ chiếm | Tắt tiến trình cũ, hoặc dùng `--port 8001` (sửa cả `--url` của app) |
| Tỉ lệ NG tăng bất thường | Đổi đèn, bẩn ống kính, lệch vị trí, hoặc sản phẩm mới chưa từng học | Kiểm tra điều kiện chụp; nếu là sản phẩm mới thì phải thu ảnh và train lại |
| Kết quả khác với lúc thử ở văn phòng | Service đang chạy `model.onnx` cũ | So `model_version` trên giao diện với `model.json`; chép lại rồi khởi động lại service |

---

## 8. Giám sát và cải tiến mô hình (vòng lặp liên tục)

Mô hình không đứng yên: sản phẩm, vật liệu, ánh sáng và camera đều thay đổi theo thời gian. Vòng cải tiến:

```
  Vận hành ──► Thu thập ảnh cần xem lại ──► Gán nhãn đúng ──► Bổ sung data/raw ──► Train lại
     ▲          (NG, HOLD, p_ng gần ngưỡng,    (người QC xác nhận)                          │
     │           ảnh bị báo nhầm/bỏ sót)                                                   ▼
     └──── Thay mô hình trên chuyền ◄── Kiểm chứng (export + compare) ◄── So sánh với mô hình cũ
```

### 8.1 Dữ liệu để theo dõi

Log của service (`predictions_<ngày>.csv`) có các cột: `timestamp`, `request_id`, `station_id`, `product_id`, `filename`, `result`, `p_ng`, `inference_ms`, `model_version`. Từ log có thể theo dõi:

- **Tỉ lệ NG theo ngày, theo trạm, theo lô**: tỉ lệ thay đổi đột ngột là dấu hiệu bất thường.
- **Số ảnh có `p_ng` gần ngưỡng** (vd từ 0,3 đến 0,7): đây là vùng mô hình "phân vân". Nếu vùng này đông lên, mô hình đang gặp loại ảnh nó chưa quen.
- **`inference_ms`**: nếu tăng dần, máy service đang quá tải.
- **`model_version`**: biết ảnh nào được phán định bằng mô hình nào, để truy vết khi có khiếu nại.

Muốn có ảnh gốc để train lại, chạy service với `--save-images`. Ảnh được lưu theo `logs/images/<ngày>/<kết quả>/`, và tên file bắt đầu bằng `request_id`, nên tra ngược từ log được. Ảnh tích lũy rất nhanh, nhớ dọn định kỳ.

### 8.2 Khi nào cần train lại

- Phát hiện NG bị lọt (khách hàng hoặc công đoạn sau báo lại).
- Tỉ lệ báo nhầm làm người kiểm tra mất thời gian quá mức.
- Thay đổi camera, ánh sáng, vị trí chụp, hoặc có mã sản phẩm mới.
- Xuất hiện loại lỗi mới chưa có trong dữ liệu.

### 8.3 Quy trình thay mô hình an toàn

1. Đổi tên `outputs/` (vd thành `outputs_v1/`) để giữ mô hình cũ, vì `train.py` sẽ ghi đè.
2. Bổ sung ảnh đã gán nhãn vào `data/raw/`, rồi chạy `split_dataset.py --overwrite` và `train.py`.
3. Chạy `evaluate.py` cho **cả mô hình mới và mô hình cũ trên cùng một tập ảnh** (mô hình cũ: `evaluate.py --checkpoint outputs_v1/best.pt`), rồi so số NG lọt và số báo nhầm. Mô hình mới chỉ được dùng nếu không tệ hơn.
4. Chạy `export.py` (phải in `Verification passed`). Giữ bản `model.onnx`/`model.json` đang chạy trên chuyền để rollback.
5. Chép sang máy service, khởi động lại service, chạy `test_client.py --compare` (phải in `Compare passed`).
6. Trên InspectionApp, bấm **Kết nối** và kiểm tra thời điểm xuất và epoch của mô hình mới.
7. Nếu có vấn đề: chép lại cặp file cũ, rồi khởi động lại service.

Phiên bản mô hình được nhận diện bằng `exported_at` và `epoch` trong `model.json`. Giá trị này xuất hiện trong mọi kết quả trả về (`model_version`) và trong log.

---

## 9. Hiện trạng và giới hạn

### 9.1 Hiện trạng

| Hạng mục | Trạng thái |
|---|---|
| Dữ liệu | Bộ thực hành **Magnetic Tile Surface Defect**: 952 OK, 392 NG (5 loại lỗi). Chỉ dùng cho học tập |
| Mô hình đang dùng | ResNet18 ở 224 pixel, epoch 10, xuất lúc 2026-10-30 14:48:54, ngưỡng NG mặc định (lớp cao nhất). Train lại sau khi gộp ảnh test cũ vào train/val |
| Mô hình cũ | `outputs_v1/`: bản đầu tiên, giữ để so sánh |
| ONNX, service, InspectionApp | Đã chạy và kiểm chứng end-to-end trên một máy |

### 9.2 Giới hạn cần biết

- **Mô hình chỉ biết sản phẩm đã học.** Đưa vào một ảnh lạ (sai sản phẩm, camera bị che), mô hình vẫn buộc phải chọn OK hoặc NG, và có thể rất tự tin. Camera phải luôn chụp đúng sản phẩm, đúng vị trí.
- **Lỗi rất nhỏ** (ví dụ Fray) có thể mất chi tiết khi ảnh bị thu về 224 pixel. Nếu loại lỗi này hay bị lọt, hãy thử tăng `IMAGE_SIZE` (train chậm hơn, suy luận chậm hơn).
- **Nhãn sai trong dữ liệu** làm mô hình học sai. Ảnh "đoán sai" nhiều khi là ảnh bị gán nhãn sai, nên luôn xem lại ảnh trước khi kết luận.
- **Service chạy 1 tiến trình** và phải bật tay. Nếu máy service khởi động lại, service không tự chạy.
- **API không có xác thực.** Chỉ dùng trong mạng nội bộ của nhà máy.
- **Chưa có vùng "cần xem lại" (REVIEW)** cho các ảnh có `p_ng` lưng chừng. Hiện tại mọi ảnh đều ra OK hoặc NG, trừ khi có lỗi kỹ thuật (HOLD).

### 9.3 Hướng phát triển tiếp

| Hạng mục | Lợi ích |
|---|---|
| Chạy service dạng Windows Service (vd bằng NSSM) | Tự khởi động cùng máy, tự chạy lại khi bị tắt |
| Thêm kết luận REVIEW cho vùng `p_ng` lưng chừng | Ảnh phân vân được đưa cho người xem, giảm cả NG lọt lẫn báo nhầm |
| Báo cáo tự động từ log (tỉ lệ NG, HOLD, phân bố `p_ng` theo ngày) | Phát hiện sớm khi điều kiện chụp thay đổi |
| Tích hợp vào phần mềm QC hiện có (WinForms) | Gọi cùng API `/predict`, áp dụng cùng nguyên tắc HOLD |
| Thay dữ liệu thực hành bằng ảnh thật của công ty | Bước bắt buộc trước khi dùng thật |

---

## 10. Tóm tắt lệnh theo giai đoạn

```
# ── Giai đoạn 1: phát triển (máy văn phòng, chạy từ thư mục gốc dự án) ──
.venv/Scripts/python.exe split_dataset.py [--overwrite]
.venv/Scripts/python.exe train.py [--resume]
.venv/Scripts/python.exe evaluate.py [--split val] [--ng-threshold 0.3]
.venv/Scripts/python.exe predict.py <ảnh|thư mục> [--show] [--only-wrong]
.venv/Scripts/python.exe export_onnx/export.py [--ng-threshold 0.3]

# ── Kiểm chứng service ──
.venv/Scripts/python.exe service/app.py
.venv/Scripts/python.exe service/test_client.py data/split/test/NG data/split/test/OK --compare

# ── Giai đoạn 2: vận hành (máy chuyền) ──
python service/app.py --host 0.0.0.0 --port 8000 [--save-images]
InspectionApp.exe --url http://<máy service>:8000 --station ST01 --watch D:\Camera
```

## Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| **Epoch** | Một lần mô hình học qua toàn bộ ảnh train |
| **Checkpoint** (`.pt`) | File lưu mô hình PyTorch |
| **ONNX** (`.onnx`) | Định dạng mô hình dùng chung, chạy bằng onnxruntime, không cần PyTorch |
| **Macro F1** | Điểm trung bình giữa các lớp, cân bằng giữa bắt đúng và báo nhầm |
| **NG recall** | Tỉ lệ ảnh NG thật mà mô hình bắt được |
| **Escape (NG lọt)** | Ảnh NG bị mô hình kết luận là OK |
| **False alarm (báo nhầm)** | Ảnh OK bị mô hình kết luận là NG |
| **HOLD** | Không phán định được vì lỗi kỹ thuật; hàng phải giữ lại cho người kiểm tra |
| **Augmentation** | Biến đổi ngẫu nhiên ảnh train (lật, xoay, đổi sáng) để mô hình học tổng quát hơn |
| **Overfit** | Mô hình "học thuộc" ảnh train, nên kém trên ảnh mới |

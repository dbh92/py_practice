# Phân loại ảnh OK/NG bằng CNN

Huấn luyện mô hình CNN để kiểm tra sản phẩm từ ảnh chụp trên dây chuyền: **OK** (đạt) hay **NG** (lỗi).

Bức tranh tổng thể (luồng dữ liệu, kiểm chứng, triển khai, vận hành hằng ngày, cải tiến mô hình) xem trong [WORKFLOW.md](WORKFLOW.md). File này là hướng dẫn lệnh chi tiết từng bước.

## Cấu trúc thư mục

```
pythonProject/
├── config.py           # MỌI thiết lập nằm ở đây (kích thước ảnh, số epoch, mô hình, ngưỡng NG...)
├── split_dataset.py    # Bước 2: chia ảnh gốc thành train / val / test
├── dataset.py          # Đọc ảnh và biến đổi ảnh (resize, augmentation, chuẩn hóa)
├── model.py            # Mô hình CNN: resnet18 (transfer learning) hoặc simple_cnn (tự viết)
├── train.py            # Bước 3: huấn luyện
├── evaluate.py         # Bước 4: đánh giá trên tập test
├── predict.py          # Bước 5: dự đoán ảnh bất kỳ
├── export_onnx/        # Bước 6: xuất mô hình sang ONNX để triển khai
│   ├── export.py           # Xuất best.pt -> model.onnx + model.json, tự kiểm chứng
│   ├── predict_onnx.py     # Dự đoán bằng ONNX, KHÔNG cần PyTorch
│   ├── requirements.txt    # Thư viện cho máy chuyền (không có torch)
│   └── model.onnx, model.json   # Do export.py tạo ra
├── service/            # Bước 7: service FastAPI phán định ảnh cho WinForms
│   ├── app.py              # Service: nhận ảnh, trả kết quả JSON
│   ├── test_client.py      # Gửi thử ảnh, kiểm chứng trùng kết quả với PyTorch
│   ├── requirements.txt    # Thư viện cho máy chuyền (không có torch)
│   └── logs/               # Do service tạo ra: log CSV từng ngày (+ ảnh nếu bật --save-images)
├── inspection_app/     # Bước 8: giao diện kiểm tra hàng (WPF, C#, .NET 8), gọi service
│   ├── MainWindow.xaml     # Bố cục giao diện
│   ├── MainWindow.xaml.cs  # Xử lý: chọn ảnh, kéo thả, theo dõi thư mục, hiện kết quả, lịch sử
│   ├── ServiceClient.cs    # Gọi API của service; lỗi nào cũng trả về HOLD
│   └── InspectionRecord.cs # Dữ liệu một lần kiểm tra và JSON của service
├── requirements.txt    # Các thư viện cần cài
├── README.md           # File hướng dẫn này
├── WORKFLOW.md         # Luồng làm việc của cả dự án và cách vận hành
├── data/
│   ├── raw/            # ← BẠN bỏ ảnh vào đây
│   │   ├── OK/
│   │   └── NG/
│   └── split/          # Do split_dataset.py tạo ra, không sửa tay
│       ├── train/OK, train/NG    (70% - để mô hình học)
│       ├── val/OK,   val/NG      (15% - để chọn epoch tốt nhất)
│       └── test/OK,  test/NG     (15% - để đánh giá sau cùng)
├── outputs/            # Do train.py / evaluate.py tạo ra
└── archive_cifar10/    # Bài CIFAR-10 cũ (giữ lại để tham khảo)
```

## Các bước thực hiện

Mọi lệnh đều chạy từ thư mục `pythonProject`.

### Bước 0: Cài đặt (chỉ khi dùng máy mới)

Máy hiện tại đã cài sẵn mọi thứ trong `.venv`. Nếu chuyển sang máy khác, hãy cài Python 3.10, rồi chạy:

```
python -m venv .venv
.venv/Scripts/pip.exe install -r requirements.txt
```

### Bước 1: Thu thập ảnh

Bỏ ảnh vào `data/raw/OK/` và `data/raw/NG/`. Tên thư mục chính là tên lớp.

> **Dữ liệu thực hành có sẵn:** `data/raw` hiện chứa bộ ảnh thật **Magnetic Tile Surface Defect**. Đây là ảnh chụp bề mặt gạch từ tính (linh kiện trong động cơ điện) trên dây chuyền: 952 ảnh OK và 392 ảnh NG, gồm 5 loại lỗi là rỗ khí (Blowhole), vỡ mẻ (Break), nứt (Crack), sờn (Fray) và không đều (Uneven). Tên file NG bắt đầu bằng loại lỗi, ví dụ `Crack_exp1_num_249594.jpg`. Nguồn và trích dẫn xem trong `data/SOURCE.txt`. Bộ ảnh chỉ dùng để học tập. Khi có ảnh thật của công ty, hãy chuyển thư mục `data/raw` cũ đi chỗ khác trước khi bỏ ảnh mới vào, để hai bộ ảnh không bị lẫn.

Những lưu ý dưới đây áp dụng khi bạn tự chụp ảnh ở công ty:

- **Chụp giống hệt lúc kiểm tra thật:** cùng camera, cùng ánh sáng, cùng góc chụp, cùng khoảng cách. Mô hình chỉ giỏi trên loại ảnh mà nó đã được học.
- **Số lượng:** nên có ít nhất khoảng 100 ảnh mỗi lớp, càng nhiều càng tốt. Số ảnh NG ít hơn OK là bình thường, vì code đã tự cân bằng.
- **NG phải đa dạng:** có đủ các loại lỗi (xước, nứt, dị vật, thiếu, lệch...), ở nhiều vị trí và nhiều mức độ khác nhau. Loại lỗi nào mô hình chưa từng thấy thì nó dễ bỏ sót.
- **Gán nhãn cẩn thận:** một ảnh NG nằm nhầm trong thư mục OK sẽ dạy sai mô hình.
- **Tránh ảnh gần như trùng nhau** (chụp cùng một sản phẩm nhiều lần liên tiếp). Nếu bản sao rơi vào cả train lẫn test, kết quả test sẽ đẹp hơn thực tế.

### Bước 2: Chia dữ liệu

```
.venv/Scripts/python.exe split_dataset.py
```

Lệnh này **sao chép** ảnh (ảnh gốc vẫn giữ nguyên) vào `data/split/`, theo tỉ lệ 70/15/15 cho từng lớp. Nếu bạn thêm ảnh mới và muốn chia lại, thêm `--overwrite`.

Kiểm tra lại kết quả chia: `.venv/Scripts/python.exe dataset.py`

### Bước 3: Huấn luyện

```
.venv/Scripts/python.exe train.py
```

Với bộ dữ liệu thực hành, máy này (CPU, không có GPU) mất khoảng 90 giây/epoch, tức tối đa khoảng 45 phút cho 30 epoch. Nếu mô hình ngừng tiến bộ sớm, chương trình sẽ tự dừng (xem bên dưới). Muốn thử nhanh trước, hãy đặt `EPOCHS = 5` trong `config.py`.

Mỗi epoch in ra một dòng:

```
Epoch 3/30 | train loss 0.2199 | val loss 0.3517 | val acc 0.9474 | val F1 0.9124 | NG recall 0.7500 *
```

- **val F1:** điểm cân bằng giữa OK và NG, dùng để chọn mô hình tốt nhất. Dấu `*` nghĩa là epoch này tốt nhất từ trước tới giờ và đã được lưu vào `outputs/best.pt`.
- **NG recall:** tỉ lệ ảnh NG bắt được. Với QC, đây là con số quan trọng nhất.
- **`(no improvement 3/10)`:** đã 3 epoch liên tiếp val F1 không tốt hơn. Khi đếm đến 10/10, chương trình **tự dừng** (early stopping) và in `Early stopping: ...`. `best.pt` vẫn là mô hình tốt nhất. Muốn luôn chạy đủ số epoch, đặt `EARLY_STOPPING_PATIENCE = None` trong `config.py`.
- Mở `outputs/training_curves.png` để xem biểu đồ. Nếu **val loss tăng trong khi train loss giảm**, mô hình đang học thuộc lòng (overfit). Lúc đó cần thêm ảnh hoặc tăng augmentation.
- Muốn dừng giữa chừng, bấm `Ctrl+C`. Mô hình tốt nhất tính đến lúc dừng vẫn còn trong `outputs/best.pt`.
- Muốn học tiếp sau khi bị ngắt, chạy `train.py --resume`. Không sửa `EPOCHS` trước khi resume.
- Chạy `train.py` (không có `--resume`) sẽ **ghi đè** kết quả cũ trong `outputs/`. Nếu muốn giữ lại kết quả cũ để so sánh, hãy đổi tên thư mục `outputs` trước khi chạy.

### Bước 4: Đánh giá trên tập test

```
.venv/Scripts/python.exe evaluate.py
```

- **QC summary:** `missed NG (escape)` là số NG bị lọt (nguy hiểm nhất), còn `false alarm` là số OK bị báo nhầm thành NG.
- `outputs/confusion_matrix_test.png`: ma trận nhầm lẫn, hàng là lớp thật, cột là lớp đoán.
- `outputs/misclassified_test.csv`: danh sách ảnh bị đoán sai (mở bằng Excel). **Hãy xem lại từng ảnh**, vì nhiều khi lỗi nằm ở nhãn sai chứ không phải ở mô hình.

### Bước 5: Dự đoán ảnh mới

```
.venv/Scripts/python.exe predict.py anh1.jpg anh2.png                     # một hoặc nhiều ảnh
.venv/Scripts/python.exe predict.py thu_muc_anh/                          # cả thư mục
.venv/Scripts/python.exe predict.py anh1.jpg --show                       # hiện ảnh kèm kết luận
.venv/Scripts/python.exe predict.py data/split/test/NG data/split/test/OK --only-wrong --show   # chỉ xem các ảnh đoán sai
.venv/Scripts/python.exe predict.py thu_muc_anh/ --save ket_qua/          # lưu lưới ảnh ra file PNG
```

Kết quả mỗi ảnh một dòng: kết luận, xác suất từng lớp, thời gian xử lý ảnh đó (mở ảnh + xử lý + dự đoán), tên file:

```
NG   | NG 82.89%  OK 17.11% |   21.3 ms | anh1.jpg
1 images | average 21.3 ms/image | slowest 21.3 ms (open + preprocess + predict)
```

Dòng cuối là tổng kết tốc độ. Trước khi bấm giờ, chương trình chạy thử mô hình một lần, vì lần chạy đầu tiên luôn chậm hơn hẳn.

- `--show`: hiện ảnh thành lưới, mỗi cửa sổ 20 ảnh. Đóng cửa sổ thì sang trang tiếp theo.
- Nếu ảnh nằm trong thư mục tên `OK` hoặc `NG` (ví dụ `data/split/test/NG/`), chương trình biết đáp án đúng. Khi đó tiêu đề có màu **xanh** nếu đoán đúng, **đỏ** nếu đoán sai, và dòng in ra có thêm `<- WRONG (true NG)`. Ảnh ở thư mục khác thì tiêu đề màu đen.
- `--only-wrong`: chỉ hiện hoặc lưu các ảnh bị đoán sai.
- Mô hình chỉ biết hai lớp OK và NG của **loại sản phẩm đã học**. Nếu đưa vào một ảnh khác hẳn (ví dụ ảnh con mèo), nó vẫn bắt buộc chọn OK hoặc NG, và có thể rất tự tin. Ngoài dây chuyền, hãy đảm bảo camera luôn chụp đúng sản phẩm và đúng vị trí.

File hỏng hoặc không phải ảnh sẽ được bỏ qua với thông báo `Skipping ...`, và chương trình tiếp tục với ảnh sau.

### Bước 6: Xuất ONNX để triển khai

ONNX là định dạng mô hình dùng chung: máy chuyền chỉ cần `onnxruntime` (nhẹ), không cần cài PyTorch.

```
.venv/Scripts/python.exe export_onnx/export.py                     # xuất outputs/best.pt và kiểm chứng trên tập test
.venv/Scripts/python.exe export_onnx/export.py --ng-threshold 0.3  # ghi ngưỡng NG vào model.json
.venv/Scripts/python.exe export_onnx/predict_onnx.py thu_muc_anh/  # dự đoán bằng ONNX, kết quả in giống predict.py
```

`export.py` làm 3 việc:

1. Ghi `export_onnx/model.onnx` (mô hình) và `export_onnx/model.json` (danh sách lớp, kích thước ảnh, mean/std, ngưỡng NG, epoch, thời điểm xuất).
2. **Kiểm chứng**: cho PyTorch và ONNX cùng dự đoán toàn bộ tập test. Đạt khi xác suất lệch nhau dưới 0,0001 và mọi kết luận trùng nhau. Không đạt thì in `Verification FAILED` và **không được dùng file đó**.
3. Đo tốc độ. Trên máy này, ResNet18 ở 224 pixel: PyTorch khoảng 19 ms, ONNX khoảng 7 ms mỗi ảnh.

**Chép sang máy chuyền**: chỉ cần 3 file `predict_onnx.py`, `model.onnx`, `model.json` và cài `pip install -r export_onnx/requirements.txt`.

Lưu ý:

- `model.onnx` chỉ chứa mạng nơ-ron. Bước xử lý ảnh (RGB, resize bilinear, chia 255, chuẩn hóa mean/std) nằm trong `predict_onnx.py`. Chương trình khác dùng `model.onnx` (ví dụ C#) phải làm **giống hệt**, đọc thông số từ `model.json`. Riêng C# và OpenCV mặc định dùng thứ tự màu BGR, phải đổi sang RGB.
- Mỗi lần train lại hoặc đổi `NG_THRESHOLD` thì phải chạy lại `export.py`.

### Bước 7: Chạy service FastAPI

WinForms gửi ảnh lên service, service trả kết quả OK/NG. Service dùng `model.onnx` và cách xử lý ảnh trong `export_onnx/`, nên kết quả giống hệt `predict_onnx.py`.

```
.venv/Scripts/pip.exe install -r service/requirements.txt          # cài một lần
.venv/Scripts/python.exe service/app.py                            # chạy service tại http://127.0.0.1:8000
.venv/Scripts/python.exe service/app.py --host 0.0.0.0 --port 8000 # cho máy khác trong mạng LAN gọi tới
.venv/Scripts/python.exe service/app.py --save-images              # lưu thêm mọi ảnh nhận được để truy vết
```

Khi service đang chạy, mở `http://127.0.0.1:8000/docs` bằng trình duyệt để xem và thử API trực tiếp.

| API | Chức năng |
|---|---|
| `POST /predict` | Gửi 1 ảnh (`multipart/form-data`, trường `file`), kèm `station_id`, `product_id` (tùy chọn). Trả về kết quả |
| `GET /health` | Kiểm tra service còn sống, mô hình đã nạp chưa |
| `GET /model-info` | Nội dung `model.json`: lớp, kích thước ảnh, ngưỡng NG, phiên bản |

Kết quả của `/predict`:

```json
{
  "request_id": "46b4a1c82eab4d1186462720393ce6cc",
  "timestamp": "2026-10-30 15:42:24.615",
  "station_id": "ST01",
  "product_id": "LOT123",
  "filename": "Fray_exp6_num_972.jpg",
  "result": "NG",
  "probabilities": {"NG": 0.733979, "OK": 0.266021},
  "p_ng": 0.733979,
  "ng_threshold": null,
  "model_version": "2026-10-30 14:48:54 (epoch 10)",
  "inference_ms": 10.7
}
```

Ảnh hỏng hoặc không phải ảnh: trả lỗi 400 (`Cannot read image: ...`). Ảnh lớn hơn 20 MB: trả lỗi 413. Service vẫn chạy tiếp.

**Log**: mỗi lần phán định ghi 1 dòng vào `service/logs/predictions_<ngày>.csv` (mở bằng Excel). Thêm `--save-images` thì lưu cả ảnh gốc vào `service/logs/images/<ngày>/<OK|NG>/`. Ảnh tích lũy rất nhanh, nhớ dọn định kỳ.

**Kiểm chứng** (mở cửa sổ thứ hai khi service đang chạy):

```
.venv/Scripts/python.exe service/test_client.py data/split/test/NG data/split/test/OK --compare
```

Script gửi từng ảnh lên service, in kết quả kèm thời gian xử lý trong service và thời gian khứ hồi. `--compare` so mọi kết quả với PyTorch (cùng cách tính như `predict.py`). Đạt khi in `Compare passed`. Trên máy này: trùng 203/203 ảnh, khứ hồi trung bình khoảng 17 ms mỗi ảnh.

**Chép sang máy chuyền**: giữ đúng cấu trúc 2 thư mục, không cần PyTorch:

```
export_onnx/predict_onnx.py, model.onnx, model.json
service/app.py, requirements.txt
```

Rồi cài `pip install -r service/requirements.txt` và chạy `python service/app.py`. Sau khi train lại: chạy `export.py`, chép `model.onnx` + `model.json` mới sang, khởi động lại service.

Lưu ý cho phía WinForms:

- Dùng **một** `HttpClient` cho cả chương trình, đặt timeout (vd 3 giây).
- **Service không trả lời, timeout hoặc báo lỗi thì xử lý như NG/HOLD và bật cảnh báo.** Tuyệt đối không cho hàng qua là OK khi chưa phán định được.
- Ghi lại `request_id` cùng sản phẩm để tra log khi cần.

### Bước 8: Giao diện kiểm tra hàng (WPF)

Chương trình Windows cho người kiểm tra: gửi ảnh lên service và hiện kết quả thật to, dễ nhìn. Viết bằng C# (.NET 8), không cần thư viện ngoài.

```
dotnet build inspection_app -c Release                                     # build (cần .NET 8 SDK)
inspection_app\bin\Release\net8.0-windows\InspectionApp.exe                # mở giao diện (service phải đang chạy)
inspection_app\bin\Release\net8.0-windows\InspectionApp.exe --url http://192.168.1.10:8000 --station ST02 --watch D:\Camera
```

Cũng có thể mở `inspection_app/InspectionApp.csproj` bằng Visual Studio 2022 rồi bấm F5.

| Phần | Chức năng |
|---|---|
| Thanh trên | Địa chỉ service, nút **Kết nối**, đèn trạng thái (cập nhật theo lần gửi ảnh gần nhất; chương trình chỉ gọi service khi có ảnh cần kiểm tra, không gọi định kỳ), thông tin mô hình, **Trạm** và **Mã SP / Lô** (gửi kèm mỗi ảnh, ghi vào log của service) |
| **Chọn ảnh...** | Chọn một hoặc nhiều ảnh để kiểm tra |
| **Kiểm tra cả thư mục...** | Kiểm tra lần lượt mọi ảnh trong một thư mục. Nút **Dừng** để dừng giữa chừng |
| **Theo dõi thư mục...** | Chọn thư mục mà phần mềm camera lưu ảnh vào: có ảnh mới là tự động kiểm tra. Chương trình chờ camera ghi xong file rồi mới đọc |
| Kéo thả | Kéo ảnh hoặc thư mục vào khung ảnh |
| Ô kết quả | **OK** (xanh), **NG** (đỏ), **HOLD** (cam). NG và HOLD có âm báo |
| Thống kê | Tổng, OK, NG, HOLD và tỉ lệ NG kể từ lúc mở chương trình |
| Chi tiết | Xác suất từng lớp, thời gian trong service, thời gian khứ hồi, ngưỡng NG, phiên bản mô hình, Request ID (chọn và copy được), lý do HOLD |
| Lịch sử | 1000 lần kiểm tra gần nhất, tô màu theo kết quả. Bấm một dòng để xem lại ảnh và kết quả. **Xuất CSV...** để lưu ra Excel |

**HOLD** nghĩa là không phán định được: service chưa chạy hoặc mất kết nối, quá 5 giây không trả lời, ảnh hỏng, hoặc không đọc được file. Hàng HOLD phải giữ lại để người kiểm tra xem, **không được cho qua**. Chương trình được viết để lỗi ở bất kỳ đâu cũng ra HOLD, không bao giờ ra OK. Lưu ý: khi service tắt hẳn, Windows mất khoảng 2 giây mới báo không kết nối được, nên HOLD hiện ra chậm hơn bình thường.

Tham số dòng lệnh (dùng khi tạo shortcut cho máy chuyền): `--url` địa chỉ service, `--station` mã trạm, `--watch` thư mục tự theo dõi ngay khi mở.

**Chép sang máy chuyền**: chép cả thư mục `inspection_app\bin\Release\net8.0-windows\`. Máy chuyền cần cài **.NET 8 Desktop Runtime** (bản Runtime, không cần SDK).

## Các file trong `outputs/`

| File | Tạo bởi | Nội dung |
|---|---|---|
| `best.pt` | `train.py` | Mô hình tốt nhất theo val F1. `evaluate.py` và `predict.py` dùng file này |
| `last.pt` | `train.py` | Trạng thái của epoch gần nhất, dùng cho `--resume` |
| `history.csv` | `train.py` | Loss và các chỉ số của từng epoch (mở bằng Excel) |
| `training_curves.png` | `train.py` | Biểu đồ loss và các chỉ số theo epoch |
| `confusion_matrix_<tập>.png` | `evaluate.py` | Ma trận nhầm lẫn của tập được đánh giá (test hoặc val) |
| `misclassified_<tập>.csv` | `evaluate.py` | Danh sách ảnh bị đoán sai, kèm xác suất |

## Ngưỡng NG: thà báo nhầm còn hơn bỏ sót

Mặc định, mô hình chọn lớp có xác suất cao nhất. Nếu muốn **bắt được nhiều NG hơn**, hãy hạ ngưỡng NG. Đổi lại, số ảnh báo nhầm sẽ tăng. Ví dụ thử nghiệm trên ảnh giả lập:

| Ngưỡng | NG bị lọt | OK bị báo nhầm |
|---|---|---|
| Mặc định (cao nhất) | 1/4 | 0/15 |
| `--ng-threshold 0.2` | **0/4** | 5/15 |

Hãy thử nhiều ngưỡng trên tập **val**, ví dụ `evaluate.py --split val --ng-threshold 0.3`. Khi đã chọn được ngưỡng, ghi nó vào `NG_THRESHOLD` trong `config.py`, rồi kiểm tra lại trên tập test.

## Các thiết lập hay chỉnh trong `config.py`

| Thiết lập | Khi nào chỉnh |
|---|---|
| `IMAGE_SIZE` | Lỗi rất nhỏ so với ảnh: tăng lên 320 hoặc 384 (train chậm hơn) |
| `HORIZONTAL_FLIP`, `VERTICAL_FLIP` | Sản phẩm có chữ hoặc hướng cố định: đặt `False` |
| `ROTATION_DEGREES`, `BRIGHTNESS`, `CONTRAST` | Chỉnh theo mức dao động thật ngoài dây chuyền |
| `MODEL_NAME` | `"resnet18"` (khuyên dùng, ít ảnh vẫn tốt) hoặc `"simple_cnn"` (cần nhiều ảnh, nhớ đổi `LEARNING_RATE = 1e-3`) |
| `EPOCHS` | Số epoch tối đa. Xem biểu đồ sau lần train đầu: nếu val F1 vẫn đang tăng ở cuối thì tăng thêm |
| `EARLY_STOPPING_PATIENCE` | Số epoch chờ trước khi tự dừng (mặc định 10). `None` = tắt. Không nên đặt dưới 5 |
| `BATCH_SIZE` | Báo lỗi hết bộ nhớ: giảm xuống 8 |
| `NUM_WORKERS` | Gặp lỗi lạ khi đọc dữ liệu: đặt về 0 |

## Mở rộng: phân loại nhiều loại lỗi

Hãy tạo thêm thư mục, ví dụ `data/raw/OK`, `data/raw/NG_xuoc`, `data/raw/NG_nut`, rồi chạy lại từ bước 2. Lưu ý: `NG recall`, `QC summary` và ngưỡng NG chỉ tính cho lớp có tên đúng bằng `NG_CLASS` trong `config.py`.

## Lỗi thường gặp

| Thông báo | Nguyên nhân và cách xử lý |
|---|---|
| `No images in: [...]` | Thư mục lớp trong `data/raw/` chưa có ảnh. Bỏ ảnh vào rồi chạy lại |
| `Need at least 2 class folders` | `data/raw/` cần ít nhất 2 thư mục lớp (vd: `OK` và `NG`) |
| `data/split already exists` | Đã chia trước đó. Thêm `--overwrite` để chia lại (ảnh gốc không bị ảnh hưởng) |
| `Warning: class '...' has too few images` | Lớp đó quá ít ảnh, nên có tập không có ảnh nào. Thu thập thêm ảnh |
| `Every class needs train images...` | Có lớp không có ảnh train, hoặc tập val trống. Thu thập thêm ảnh rồi chia lại |
| `Checkpoint not found` | Chưa train. Chạy `train.py` trước |
| `model.json not found in ...` | Chưa xuất ONNX. Chạy `export_onnx/export.py` trước, hoặc chỉ đúng thư mục bằng `--model-dir` |
| `No images in data/split/test. Use --split val.` | Tập test trống (vd `TEST_RATIO = 0`). Kiểm chứng bằng tập val: `export.py --split val` |
| `Verification FAILED: do not deploy this model.onnx.` | ONNX cho kết quả khác PyTorch. Không dùng file vừa xuất; báo lại để kiểm tra |
| `Cannot connect to http://127.0.0.1:8000. Start the service first: ...` | Service chưa chạy hoặc sai địa chỉ. Chạy `service/app.py` ở một cửa sổ khác, hoặc sửa `--url` |
| `Compare FAILED. If you retrained, run export_onnx/export.py again, then restart the service.` | Service khác PyTorch. Thường do train lại mà chưa xuất lại ONNX, hoặc chưa khởi động lại service |
| `[Errno 10048] error while attempting to bind on address` khi chạy service | Cổng 8000 đang bị chương trình khác dùng (thường là service cũ chưa tắt). Tắt nó, hoặc dùng `--port 8001` |
| `Class names changed ..., cannot resume` | Đã thêm hoặc đổi tên thư mục lớp sau lần train trước. Chạy `train.py` không có `--resume` |
| `out of memory` / máy rất chậm | Giảm `BATCH_SIZE` hoặc `IMAGE_SIZE` trong `config.py` |
| Lỗi liên quan đến `DataLoader` / worker | Đặt `NUM_WORKERS = 0` trong `config.py` |
| `URLError` khi train lần đầu | Máy không tải được trọng số pretrained. Xem mục bên dưới |

## Máy không có internet

Lần đầu dùng `resnet18`, chương trình cần tải trọng số pretrained (khoảng 45 MB). Trên máy không có mạng, hãy chép file `resnet18-*.pth` từ thư mục `C:\Users\<tên>\.cache\torch\hub\checkpoints\` của máy đã tải sang đúng thư mục đó trên máy mới.

# Phân loại ảnh OK/NG bằng CNN

Huấn luyện mô hình CNN để kiểm tra sản phẩm từ ảnh chụp trên dây chuyền: **OK** (đạt) hay **NG** (lỗi).

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
├── requirements.txt    # Các thư viện cần cài
├── README.md           # File hướng dẫn này
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

Kết quả: `NG   | NG 82.89%  OK 17.11% | anh1.jpg`

- `--show`: hiện ảnh thành lưới, mỗi cửa sổ 20 ảnh. Đóng cửa sổ thì sang trang tiếp theo.
- Nếu ảnh nằm trong thư mục tên `OK` hoặc `NG` (ví dụ `data/split/test/NG/`), chương trình biết đáp án đúng. Khi đó tiêu đề có màu **xanh** nếu đoán đúng, **đỏ** nếu đoán sai, và dòng in ra có thêm `<- WRONG (true NG)`. Ảnh ở thư mục khác thì tiêu đề màu đen.
- `--only-wrong`: chỉ hiện hoặc lưu các ảnh bị đoán sai.
- Mô hình chỉ biết hai lớp OK và NG của **loại sản phẩm đã học**. Nếu đưa vào một ảnh khác hẳn (ví dụ ảnh con mèo), nó vẫn bắt buộc chọn OK hoặc NG, và có thể rất tự tin. Ngoài dây chuyền, hãy đảm bảo camera luôn chụp đúng sản phẩm và đúng vị trí.

File hỏng hoặc không phải ảnh sẽ được bỏ qua với thông báo `Skipping ...`, và chương trình tiếp tục với ảnh sau.

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
| `Class names changed ..., cannot resume` | Đã thêm hoặc đổi tên thư mục lớp sau lần train trước. Chạy `train.py` không có `--resume` |
| `out of memory` / máy rất chậm | Giảm `BATCH_SIZE` hoặc `IMAGE_SIZE` trong `config.py` |
| Lỗi liên quan đến `DataLoader` / worker | Đặt `NUM_WORKERS = 0` trong `config.py` |
| `URLError` khi train lần đầu | Máy không tải được trọng số pretrained. Xem mục bên dưới |

## Máy không có internet

Lần đầu dùng `resnet18`, chương trình cần tải trọng số pretrained (khoảng 45 MB). Trên máy không có mạng, hãy chép file `resnet18-*.pth` từ thư mục `C:\Users\<tên>\.cache\torch\hub\checkpoints\` của máy đã tải sang đúng thư mục đó trên máy mới.

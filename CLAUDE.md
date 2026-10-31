# CLAUDE.md

File này cung cấp hướng dẫn cho Claude Code (claude.ai/code) khi làm việc với code trong repository này.

## Tổng quan

Dự án phân loại ảnh OK/NG bằng CNN, dùng ảnh tự thu thập từ dây chuyền sản xuất (công việc QC). Người dùng là người mới học PyTorch. Dự án không có test hay linter. Git remote là `https://github.com/dbh92/py_practice.git`, file `*.pt` và `*.onnx` lưu bằng Git LFS. `README.md` là hướng dẫn sử dụng cho người dùng, viết bằng tiếng Việt: khi đổi cách dùng, thiết lập trong config hoặc file trong `outputs/`, phải cập nhật cả README. Bảng "Lỗi thường gặp" trong README trích **nguyên văn** các câu `print` báo lỗi, nên khi sửa câu thông báo thì phải sửa cả bảng đó. `WORKFLOW.md` (tiếng Việt) mô tả luồng tổng thể, chuỗi kiểm chứng, triển khai và vận hành, kèm số liệu đo được: khi đổi kiến trúc, quy tắc HOLD, cách kết nối hay mô hình đang dùng thì cập nhật cả file này. `data/split/` và `outputs/` do script sinh ra, không sửa tay; `train.py` chạy không có `--resume` sẽ ghi đè `outputs/`. `img.png` ở thư mục gốc là file của người dùng, đừng xóa hay di chuyển. `archive_cifar10/` là bài tập CIFAR-10 cũ (gồm MLP, CNN và dữ liệu riêng), độc lập với dự án hiện tại: đừng sửa hay import từ đó.

Mỗi dòng code đều có **comment tiếng Việt cuối dòng**, giải thích dòng đó làm gì và vì sao. Khi thêm hoặc sửa code, hãy giữ phong cách này. Các câu `print` viết bằng tiếng Anh, vì stdout trên máy này có thể dùng cp1252, và in tiếng Việt có dấu sẽ gây `UnicodeEncodeError`. Khi chạy thử qua pipe, hãy đặt `PYTHONIOENCODING=utf-8`, vì tên file ảnh có thể có dấu.

## Lệnh

Python 3.10 trong `.venv` (xem `requirements.txt`). Máy không có GPU. Chạy mọi lệnh từ thư mục gốc, vì đường dẫn trong `config.py` là tương đối.

```
.venv/Scripts/python.exe split_dataset.py [--overwrite]   # data/raw/<lớp>/ -> data/split/{train,val,test}/<lớp>/
.venv/Scripts/python.exe dataset.py                       # in số ảnh mỗi tập/lớp, kiểm tra một batch
.venv/Scripts/python.exe model.py                         # kiểm tra forward của các mô hình
.venv/Scripts/python.exe train.py [--resume]
.venv/Scripts/python.exe evaluate.py [--split val] [--ng-threshold 0.3]
.venv/Scripts/python.exe predict.py <ảnh|thư mục>... [--show] [--save DIR] [--only-wrong] [--ng-threshold 0.3]
.venv/Scripts/python.exe export_onnx/export.py [--checkpoint P] [--split val] [--ng-threshold 0.3]
.venv/Scripts/python.exe export_onnx/predict_onnx.py <ảnh|thư mục>... [--model-dir DIR] [--ng-threshold 0.3]
.venv/Scripts/python.exe service/app.py [--host 0.0.0.0] [--port 8000] [--model-dir DIR] [--log-dir DIR] [--save-images]
.venv/Scripts/python.exe service/test_client.py <ảnh|thư mục>... [--url URL] [--compare]   # service phải đang chạy
dotnet build inspection_app -c Release        # giao diện WPF (.NET 8 SDK đã cài sẵn, có cả VS 2022)
inspection_app/bin/Release/net8.0-windows/InspectionApp.exe [--url URL] [--station ST01] [--watch DIR]
```

`data/raw/` hiện chứa bộ dữ liệu thực hành **Magnetic Tile Surface Defect**: 952 ảnh OK và 392 ảnh NG, đều là ảnh xám, kích thước không đồng đều (từ 103 đến 632 pixel). Tên file NG có tiền tố loại lỗi: `Blowhole_`, `Break_`, `Crack_`, `Fray_`, `Uneven_`. Nguồn, trích dẫn và cách sắp xếp ghi trong `data/SOURCE.txt`; dataset chỉ dùng cho học tập và nghiên cứu. Muốn chạy thử toàn bộ quy trình, hãy chép `*.py` và `data/raw` sang thư mục tạm, đặt `EPOCHS` nhỏ bằng `sed`, rồi chạy ở thư mục tạm, để `data/split` và `outputs/` của người dùng không bị ghi đè. Với dataset này (940 ảnh train), ResNet18 ở 224 pixel chạy khoảng 90 giây/epoch trên CPU.

## Kiến trúc

- `config.py`: **nơi duy nhất** chứa thiết lập (đường dẫn, tỉ lệ chia, kích thước ảnh, mô hình, siêu tham số, augmentation, `NG_CLASS`, `NG_THRESHOLD`). Các script khác chỉ đọc từ đây; chỉ có những thao tác theo từng lần chạy mới là tham số dòng lệnh.
- **Tên lớp = tên thư mục**, sắp xếp theo alphabet (`['NG', 'OK']`), nên nhãn số là chỉ số trong danh sách đó. `get_class_names()` đọc từ `data/split/train`. Danh sách lớp được lưu trong checkpoint, và `evaluate.py`/`predict.py` dùng danh sách từ checkpoint chứ không đọc lại thư mục.
- `split_dataset.py`: chia riêng từng lớp (stratified), dùng `random.Random(config.SEED)`, và **sao chép** chứ không di chuyển ảnh. Script từ chối chạy nếu có lớp trống, hoặc nếu `data/split` đã tồn tại mà không có `--overwrite`.
- `dataset.py`: `ImageClassificationDataset` chỉ lưu danh sách `(đường dẫn, nhãn)` và mở ảnh bằng PIL `convert("RGB")` trong `__getitem__`. `build_transforms(train, image_size)` gồm Resize, augmentation (chỉ khi train, bật tắt theo config), ToTensor, rồi Normalize theo mean/std của ImageNet. Thứ tự `dataset.samples` khớp với thứ tự output khi `shuffle=False`; `evaluate.py` dựa vào điều này để ghép ảnh với dự đoán.
- `model.py`: `build_model(name, num_classes, pretrained)` hỗ trợ `"resnet18"` (torchvision, thay lớp `fc`) và `"simple_cnn"` (dùng `AdaptiveAvgPool2d`, nên chạy được với mọi `IMAGE_SIZE`). `load_model_from_checkpoint()` dựng lại mô hình từ `model_name` lưu trong checkpoint. Checkpoint là dict gồm `model_name`, `class_names`, `image_size`, `epoch`, `model`; `last.pt` có thêm `optimizer`, `scheduler`, `best_score`, `history`.
- `evaluate.py` vừa là script, vừa là **module dùng chung**: `run_inference` (trả về xác suất softmax), `decide_predictions` (argmax, hoặc dùng ngưỡng xác suất NG riêng), `compute_metrics` (accuracy, macro F1, recall của NG). `train.py` và `predict.py` import các hàm này, nên quy tắc ra kết luận ở mọi nơi là một. `matplotlib.use("Agg")` chỉ được gọi trong `__main__` của `evaluate.py`; nếu gọi ở đầu file thì `predict.py --show` sẽ không mở được cửa sổ.
- `train.py`: dùng `CrossEntropyLoss` với trọng số lớp `sum/(k*count)` khi `USE_CLASS_WEIGHTS` bật, AdamW và CosineAnnealingLR. Mô hình tốt nhất được chọn theo **macro F1 trên validation** (không theo accuracy, vì NG thường rất ít). Sau mỗi epoch, script ghi `outputs/best.pt`, `last.pt`, `history.csv` và `training_curves.png`. Resume yêu cầu `class_names` không đổi và `EPOCHS` giữ nguyên, vì `T_max` của scheduler được nạp từ checkpoint. Early stopping: biến `epochs_without_improvement` được reset về 0 khi val F1 đạt mức tốt nhất mới, và được lưu trong `last.pt`. Điều kiện dừng (`>= EARLY_STOPPING_PATIENCE`) được kiểm tra ở **đầu** vòng lặp epoch, nên `--resume` sau khi đã tự dừng sẽ thoát ngay. `None` tắt tính năng này. Patience nhỏ dễ dừng quá sớm, vì với cosine LR các epoch cuối vẫn thường cải thiện.
- `predict.py`: nhãn thật được suy ra từ **tên thư mục cha** của ảnh, nếu tên đó trùng một lớp (`true_label_from_path`). Nhãn này chỉ dùng để tô xanh/đỏ, đánh dấu `<- WRONG` và lọc `--only-wrong`. `--show`/`--save` vẽ ảnh thu nhỏ (tối đa 256 pixel) thành lưới `IMAGES_PER_PAGE = 20`. `plt.show()` sẽ chặn chương trình, nên khi chạy thử không có màn hình, hãy dùng `MPLBACKEND=Agg` với `--save <thư mục tạm>` rồi xem file PNG.
- `export_onnx/` (bước triển khai):
  - `predict_onnx.py` chứa `OnnxClassifier` và **cố ý không import torch, `config` hay file nào của dự án**, để chép riêng 3 file (`predict_onnx.py`, `model.onnx`, `model.json`) sang máy chuyền. Mọi thông số đọc từ `model.json`. Hàm `preprocess` (PIL + numpy) phải khớp với `build_transforms(train=False)`, và `decide` phải khớp với `decide_predictions`. Sửa một bên thì sửa cả bên kia. `IMAGE_EXTENSIONS` cũng được chép lại từ `config.py`.
  - `export.py` thêm thư mục gốc vào `sys.path` để import các module của dự án, nên phải chạy từ thư mục gốc. Script dùng exporter dynamo của torch (cần `onnxscript`), đặt `external_data=False` để gộp thành 1 file, và `verbose=False` vì exporter in emoji gây `UnicodeEncodeError` trên cp1252. Batch là chiều động.
  - `export.py` kiểm chứng `OnnxClassifier` (đúng đường chạy trên máy chuyền) so với `run_inference` của PyTorch. Đạt khi chênh lệch xác suất nhỏ hơn 1e-4 và kết luận trùng 100%; nếu không, thoát với mã 1.
  - `model.onnx` và `model.json` là file sinh ra. `*.onnx` được lưu bằng Git LFS.
  - `checkpoint_label()` ghi đường dẫn checkpoint vào `model.json`. Nếu file nằm trong dự án thì ghi đường dẫn tương đối; nếu nằm ngoài dự án hoặc ở ổ đĩa khác thì ghi đường dẫn đầy đủ, vì trên Windows `os.path.relpath` báo `ValueError` khi hai đường dẫn khác ổ.
- `service/` (FastAPI, không dùng torch):
  - `app.py` thêm `../export_onnx` vào `sys.path` để import `OnnxClassifier`, nên **không chép lại** code xử lý ảnh. Khi triển khai phải giữ cấu trúc 2 thư mục cạnh nhau. Mô hình được nạp một lần trong `lifespan`, rồi chạy thử với một ảnh đen.
  - Endpoint `/predict` khai báo bằng `def`, không dùng `async def`, để phần suy luận nặng CPU chạy trong threadpool. Thiết lập lưu trong dict `settings`, ghi đè bằng argparse trong `__main__`, rồi gọi `uvicorn.run(app)` với 1 tiến trình.
  - Log ghi vào `logs/predictions_<ngày>.csv` (utf-8-sig, có `threading.Lock`). Nếu bật `--save-images`, ảnh gốc được lưu vào `logs/images/<ngày>/<lớp>/`. `service/logs/` nằm trong `.gitignore`.
  - `test_client.py` chỉ import torch khi có `--compare`, và khi so sánh dùng ngưỡng NG lấy từ `/model-info` của service. Khi chạy thử, dùng `--log-dir` trỏ vào thư mục tạm để không ghi log vào dự án. Muốn tắt service chạy nền, dừng tiến trình đang chiếm cổng 8000.
- `inspection_app/` (WPF, C#, `net8.0-windows`, không dùng gói NuGet; `ImplicitUsings` và `Nullable` đều tắt, nên phải viết `using` tường minh):
  - Code C# cũng có comment tiếng Việt cuối mỗi dòng. Chữ trên giao diện viết bằng tiếng Việt, vì WPF không gặp vấn đề cp1252.
  - Cấu trúc theo kiểu code-behind, không dùng MVVM, để người mới dễ đọc.
  - `ServiceClient` dùng một `HttpClient` tĩnh, timeout 5 giây, và **không đặt `BaseAddress`**: URL đổi được trên giao diện, nên mỗi lần gọi đều ghép `BaseUrl` với đường dẫn API. Client đặt `ConnectionClose = true` (không giữ kết nối) và không gọi `/health` định kỳ: uvicorn đóng kết nối rảnh sau 5 giây, nên nếu dùng lại kết nối cũ thì thỉnh thoảng gặp lỗi "connection forcibly closed" và ra HOLD giả. Đèn trạng thái cập nhật theo `InspectionRecord.ServiceReached` của lần gửi gần nhất.
  - **`InspectAsync` không bao giờ ném lỗi**: lỗi nào cũng thành `Result = "HOLD"` kèm `Error`. `InspectionRecord.Result` mặc định là HOLD. Không được làm yếu quy tắc an toàn này.
  - Các lần kiểm tra được xếp hàng tuần tự bằng `SemaphoreSlim(1)`. Sự kiện của `FileSystemWatcher` (gồm cả Created và Renamed) chạy ở luồng nền, nên phải đưa về luồng giao diện qua `Dispatcher.InvokeAsync`, rồi chờ tới khi mở file được với `FileShare.None` mới gửi đi.
  - Kết quả không phải "OK" và không phải HOLD đều được đếm và tô màu như NG, để vẫn đúng khi mô hình có nhiều lớp lỗi.
  - Muốn thử tự động không cần bấm tay: bật service, chạy exe với `--watch <thư mục tạm>`, chép ảnh vào thư mục đó, rồi chụp cửa sổ bằng PowerShell (`SetProcessDPIAware` + `GetWindowRect` + `CopyFromScreen`).
- Các chỉ số dành riêng cho QC (NG recall, `QC summary` gồm số NG bị lọt và số báo nhầm, ngưỡng NG) chỉ có tác dụng khi tồn tại lớp có tên đúng bằng `config.NG_CLASS`.

## Ràng buộc

- Windows và `NUM_WORKERS > 0` nghĩa là worker được tạo bằng spawn: code chạy ở cấp module phải nằm trong `if __name__ == '__main__':`, và dataset phải pickle được.
- Trọng số ResNet18 pretrained đã có trong cache (`C:\Users\Wisol\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth`). Trên máy offline phải chép file này sang, hoặc đặt `PRETRAINED = False`.
- Gói `python-opencv2` trong `.venv` không phải OpenCV, mà là một bộ bài lab mẫu, và dự án không dùng tới nó.

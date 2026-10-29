# Mọi thiết lập của dự án nằm ở đây. Muốn đổi gì thì sửa file này, không cần sửa các file khác.

# ===== Đường dẫn =====
RAW_DIR = "data/raw"  # Ảnh gốc bạn thu thập, mỗi lớp một thư mục: data/raw/OK/*.jpg, data/raw/NG/*.jpg
SPLIT_DIR = "data/split"  # split_dataset.py chia ảnh gốc vào đây thành train/ val/ test/
OUTPUT_DIR = "outputs"  # Nơi lưu checkpoint, biểu đồ, báo cáo sau khi train/đánh giá

# ===== Chia dữ liệu =====
VAL_RATIO = 0.15  # 15% ảnh mỗi lớp dùng làm validation (chọn epoch tốt nhất trong lúc train)
TEST_RATIO = 0.15  # 15% ảnh mỗi lớp dùng làm test (chỉ dùng để đánh giá sau cùng); 70% còn lại để train
SEED = 42  # Hạt giống ngẫu nhiên: giữ nguyên thì lần nào chia dữ liệu/train cũng ra kết quả giống nhau

# ===== Ảnh =====
IMAGE_SIZE = 224  # Mọi ảnh được thu nhỏ về 224x224 (kích thước chuẩn của ResNet); lỗi rất nhỏ thì thử tăng lên 320 hoặc 384
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")  # Các đuôi file được coi là ảnh
MEAN = (0.485, 0.456, 0.406)  # Mean theo kênh R, G, B của ImageNet (bộ ảnh mà ResNet pretrained đã học), dùng để chuẩn hóa
STD = (0.229, 0.224, 0.225)  # Độ lệch chuẩn theo kênh R, G, B của ImageNet

# ===== Mô hình =====
MODEL_NAME = "resnet18"  # "resnet18": CNN có sẵn, dùng transfer learning (khuyên dùng); "simple_cnn": CNN tự viết, học từ đầu
PRETRAINED = True  # True: dùng trọng số đã học trên ImageNet (cần tải về lần đầu, khoảng 45 MB); False: khởi tạo ngẫu nhiên

# ===== Huấn luyện =====
EPOCHS = 30  # Số lần duyệt hết tập train
BATCH_SIZE = 16  # Số ảnh mỗi lần học; giảm xuống 8 nếu báo hết bộ nhớ
LEARNING_RATE = 1e-4  # Tốc độ học; 1e-4 hợp với resnet18 pretrained, nếu dùng simple_cnn nên tăng lên 1e-3
WEIGHT_DECAY = 1e-4  # Phạt trọng số lớn để chống học thuộc lòng (overfit)
NUM_WORKERS = 4  # Số tiến trình phụ đọc và xử lý ảnh song song (máy này có 8 nhân CPU); gặp lỗi thì đặt về 0
USE_CLASS_WEIGHTS = True  # True: lớp có ít ảnh (thường là NG) được tính loss nặng hơn để mô hình không bỏ qua nó
EARLY_STOPPING_PATIENCE = 10  # Tự dừng khi val F1 không tốt hơn trong 10 epoch liên tiếp; None = tắt, luôn chạy đủ EPOCHS. Không nên đặt quá nhỏ (< 5) vì learning rate giảm dần nên các epoch cuối vẫn có thể cải thiện

# ===== Augmentation (biến đổi ngẫu nhiên ảnh train để mô hình học tổng quát hơn) =====
# Chỉ bật phép biến đổi nào mà ảnh THẬT ngoài dây chuyền cũng có thể như vậy
HORIZONTAL_FLIP = True  # Lật ngang; tắt (False) nếu sản phẩm có chữ/ký hiệu hoặc hướng cố định
VERTICAL_FLIP = True  # Lật dọc; tắt nếu sản phẩm luôn nằm cùng một chiều
ROTATION_DEGREES = 10  # Xoay ngẫu nhiên trong khoảng ±10 độ (sản phẩm đặt hơi lệch); 0 = không xoay
BRIGHTNESS = 0.2  # Thay đổi độ sáng ngẫu nhiên ±20% (ánh sáng dây chuyền dao động); 0 = tắt
CONTRAST = 0.2  # Thay đổi độ tương phản ngẫu nhiên ±20%; 0 = tắt

# ===== Kết luận OK/NG =====
NG_CLASS = "NG"  # Tên thư mục của lớp NG (phải trùng với tên trong data/raw)
NG_THRESHOLD = None  # None: chọn lớp có xác suất cao nhất. Đặt số (vd 0.3): xác suất NG >= 0.3 là kết luận NG -> bắt được nhiều NG hơn, đổi lại báo nhầm nhiều hơn

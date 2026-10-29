import torch  # Thư viện PyTorch chính
import torch.nn as nn  # Module chứa các lớp mạng nơ-ron
from torchvision import models  # Các mô hình CNN có sẵn (ResNet, EfficientNet, ...)

MODEL_NAMES = ("resnet18", "simple_cnn")  # Các tên mô hình hợp lệ cho config.MODEL_NAME


def conv_block(in_channels, out_channels):  # Một khối tích chập: Conv -> BatchNorm -> ReLU -> MaxPool
    return nn.Sequential(  # Xếp các lớp nối tiếp
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Bộ lọc 3x3 quét ảnh tìm đặc trưng (cạnh, vết, hoa văn)
        nn.BatchNorm2d(out_channels),  # Chuẩn hóa theo batch: học ổn định và nhanh hơn
        nn.ReLU(),  # Hàm kích hoạt: số âm thành 0
        nn.MaxPool2d(kernel_size=2),  # Giữ giá trị lớn nhất trong mỗi ô 2x2 -> ảnh nhỏ đi một nửa
    )


class SimpleCNN(nn.Module):  # CNN tự viết, học từ đầu; cần nhiều ảnh hơn resnet18 pretrained
    def __init__(self, num_classes):  # num_classes: số lớp (vd 2 cho OK/NG)
        super().__init__()  # Gọi hàm khởi tạo của nn.Module (bắt buộc)
        self.features = nn.Sequential(  # Phần trích xuất đặc trưng: 4 khối, ảnh 224 -> 112 -> 56 -> 28 -> 14
            conv_block(3, 32),  # 3 kênh màu -> 32 kênh đặc trưng
            conv_block(32, 64),  # 32 -> 64
            conv_block(64, 128),  # 64 -> 128
            conv_block(128, 256),  # 128 -> 256
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # Lấy trung bình mỗi kênh về 1 số -> chạy được với mọi kích thước ảnh (IMAGE_SIZE nào cũng được)
        self.classifier = nn.Sequential(  # Phần phân loại
            nn.Flatten(),  # (batch, 256, 1, 1) -> (batch, 256)
            nn.Dropout(0.3),  # Khi train, tắt ngẫu nhiên 30% nơ-ron để chống học thuộc lòng
            nn.Linear(256, num_classes),  # 256 -> số lớp: điểm số (logits) của từng lớp, không có ReLU ở cuối
        )

    def forward(self, x):  # Ảnh đi qua mô hình theo thứ tự nào
        x = self.features(x)  # Trích xuất đặc trưng
        x = self.pool(x)  # Gom về 1 số mỗi kênh
        return self.classifier(x)  # Trả về logits (batch, num_classes)


def build_model(model_name, num_classes, pretrained=True):  # Tạo mô hình theo tên
    if model_name == "simple_cnn":  # CNN tự viết
        return SimpleCNN(num_classes)  # Không có trọng số pretrained, tham số pretrained bị bỏ qua
    if model_name == "resnet18":  # ResNet18 có sẵn của torchvision
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None  # Trọng số đã học trên ImageNet (1,2 triệu ảnh) hoặc ngẫu nhiên
        model = models.resnet18(weights=weights)  # Tạo ResNet18 (lần đầu sẽ tải trọng số về ~/.cache/torch)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # Transfer learning: thay lớp cuối 1000 lớp ImageNet bằng lớp cuối mới cho số lớp của bạn
        return model  # Trả về mô hình
    raise ValueError("Unknown model '{}', choose one of {}".format(model_name, MODEL_NAMES))  # Tên sai thì báo lỗi


def load_model_from_checkpoint(checkpoint_path, device):  # Nạp mô hình đã train từ file checkpoint (dùng cho evaluate.py, predict.py)
    checkpoint = torch.load(checkpoint_path, map_location=device)  # Đọc file; map_location đưa tensor về đúng thiết bị (CPU/GPU)
    model = build_model(checkpoint["model_name"], len(checkpoint["class_names"]), pretrained=False)  # Tạo mô hình rỗng cùng kiến trúc (không cần tải pretrained vì sẽ nạp trọng số ngay sau)
    model.load_state_dict(checkpoint["model"])  # Nạp trọng số đã học
    model.to(device)  # Chuyển mô hình sang thiết bị
    model.eval()  # Chế độ đánh giá: tắt Dropout, BatchNorm dùng thống kê đã học
    return model, checkpoint  # Trả về mô hình và thông tin đi kèm (class_names, image_size, ...)


if __name__ == '__main__':  # Chạy trực tiếp file để kiểm tra các mô hình
    input_data = torch.rand(2, 3, 224, 224)  # 2 ảnh giả ngẫu nhiên kích thước 3x224x224
    for name in MODEL_NAMES:  # Với từng mô hình
        model = build_model(name, num_classes=2)  # Tạo mô hình 2 lớp (OK/NG)
        num_params = sum(p.numel() for p in model.parameters())  # Đếm tổng số tham số
        print("{:<11} output {} | {:,} params".format(name, tuple(model(input_data).shape), num_params))  # Output phải là (2, 2)

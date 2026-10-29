import torch  # Thư viện PyTorch chính
import torch.nn as nn  # Module chứa các lớp mạng nơ-ron (Linear, Conv2d, ReLU, ...)


class SimpleNeuralNetwork(nn.Module):  # Mạng MLP (chỉ gồm các lớp kết nối đầy đủ), kế thừa nn.Module như mọi mô hình PyTorch
    def __init__(self, num_classes=10):  # num_classes: số lớp cần phân loại (CIFAR-10 có 10 lớp)
        super().__init__()  # Gọi hàm khởi tạo của nn.Module (bắt buộc)
        self.flatten = nn.Flatten()  # Duỗi phẳng ảnh (3, 32, 32) thành vector 3072 phần tử
        # Lớp cuối trả về logits thô, không dùng ReLU vì CrossEntropyLoss cần cả giá trị âm
        self.fc = nn.Sequential(  # Sequential: xếp các lớp nối tiếp, dữ liệu đi lần lượt qua từng lớp
            nn.Linear(in_features=3 * 32 * 32, out_features=256),  # Lớp kết nối đầy đủ: 3072 đầu vào -> 256 đầu ra
            nn.ReLU(),  # Hàm kích hoạt ReLU: max(0, x), giúp mạng học được quan hệ phi tuyến
            nn.Linear(in_features=256, out_features=512),  # 256 -> 512
            nn.ReLU(),  # Kích hoạt
            nn.Linear(in_features=512, out_features=1024),  # 512 -> 1024
            nn.ReLU(),  # Kích hoạt
            nn.Linear(in_features=1024, out_features=512),  # 1024 -> 512
            nn.ReLU(),  # Kích hoạt
            nn.Linear(in_features=512, out_features=num_classes),  # 512 -> 10: mỗi đầu ra là điểm số (logit) của một lớp
        )

    def forward(self, x):  # Hàm lan truyền xuôi: định nghĩa dữ liệu đi qua mạng thế nào; x có kích thước (batch, 3, 32, 32)
        x = self.flatten(x)  # (batch, 3, 32, 32) -> (batch, 3072)
        x = self.fc(x)  # (batch, 3072) -> (batch, 10)
        return x  # Trả về logits, mỗi hàng là 10 điểm số của một ảnh


class SimpleCNN(nn.Module):  # Mạng tích chập (CNN): giữ được cấu trúc không gian của ảnh nên chính xác hơn MLP nhiều
    def __init__(self, num_classes=10):  # num_classes: số lớp cần phân loại
        super().__init__()  # Gọi hàm khởi tạo của nn.Module
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.features = nn.Sequential(  # Phần trích xuất đặc trưng: 3 khối tích chập nối tiếp
            self._conv_block(3, 32),  # Khối 1: 3 kênh màu -> 32 kênh đặc trưng, ảnh 32x32 -> 16x16
            self._conv_block(32, 64),  # Khối 2: 32 -> 64 kênh, ảnh 16x16 -> 8x8
            self._conv_block(64, 128),  # Khối 3: 64 -> 128 kênh, ảnh 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(  # Phần phân loại: dựa trên đặc trưng để đưa ra điểm số cho từng lớp
            nn.Flatten(),  # Duỗi phẳng (batch, 128, 4, 4) -> (batch, 2048)
            nn.Dropout(0.5),  # Khi train, tắt ngẫu nhiên 50% nơ-ron để chống overfit (khi eval tự động tắt)
            nn.Linear(in_features=128 * 4 * 4, out_features=256),  # 2048 -> 256
            nn.ReLU(),  # Kích hoạt
            nn.Dropout(0.5),  # Dropout thêm một lần nữa
            nn.Linear(in_features=256, out_features=num_classes),  # 256 -> 10 logits
        )

    @staticmethod  # Hàm tĩnh: không cần self, chỉ là hàm tiện ích tạo khối
    def _conv_block(in_channels, out_channels):  # Tạo một khối tích chập: 2 lần (Conv -> BatchNorm -> ReLU) rồi MaxPool
        return nn.Sequential(  # Xếp các lớp nối tiếp
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Tích chập 3x3; padding=1 giữ nguyên kích thước ảnh
            nn.BatchNorm2d(out_channels),  # Chuẩn hóa theo batch: ổn định và tăng tốc quá trình học
            nn.ReLU(),  # Kích hoạt
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Tích chập 3x3 lần hai, giữ số kênh
            nn.BatchNorm2d(out_channels),  # Chuẩn hóa theo batch
            nn.ReLU(),  # Kích hoạt
            nn.MaxPool2d(kernel_size=2),  # Lấy giá trị lớn nhất trong mỗi ô 2x2 -> chiều cao và chiều rộng giảm một nửa
        )

    def forward(self, x):  # Lan truyền xuôi; x có kích thước (batch, 3, 32, 32)
        x = self.features(x)  # Trích xuất đặc trưng: -> (batch, 128, 4, 4)
        x = self.classifier(x)  # Phân loại: -> (batch, 10)
        return x  # Trả về logits


MODELS = {  # Bảng tra tên -> lớp mô hình; train_neuralnetwork.py dùng bảng này cho tham số --model
    "mlp": SimpleNeuralNetwork,  # --model mlp
    "cnn": SimpleCNN,  # --model cnn
}


if __name__ == '__main__':  # Chỉ chạy khi chạy trực tiếp file (python models.py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Dùng GPU nếu có, không thì dùng CPU
    input_data = torch.rand(8, 3, 32, 32, device=device)  # Tạo dữ liệu giả: 8 ảnh ngẫu nhiên kích thước 3x32x32
    for name, model_class in MODELS.items():  # Duyệt qua từng mô hình trong bảng
        model = model_class().to(device)  # Tạo mô hình và chuyển sang thiết bị (CPU/GPU)
        num_params = sum(p.numel() for p in model.parameters())  # Đếm tổng số tham số (trọng số) của mô hình
        print(name, model(input_data).shape, "{:,} params".format(num_params))  # In tên, kích thước output (phải là [8, 10]) và số tham số

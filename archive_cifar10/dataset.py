import os  # Thư viện làm việc với đường dẫn file/thư mục của hệ điều hành
import pickle  # Thư viện đọc/ghi đối tượng Python đã được "đóng gói" (CIFAR-10 lưu dữ liệu theo dạng này)
import numpy as np  # Thư viện tính toán mảng số học
import torch  # Thư viện PyTorch chính
from torch.utils.data import Dataset, DataLoader  # Dataset: lớp cơ sở để định nghĩa bộ dữ liệu; DataLoader: chia dữ liệu thành batch
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Normalize  # Các phép biến đổi ảnh

# Mean/std của tập train CIFAR-10, theo từng kênh RGB
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)  # Giá trị trung bình của kênh R, G, B (tính trên thang [0, 1])
CIFAR_STD = (0.2470, 0.2435, 0.2616)  # Độ lệch chuẩn của kênh R, G, B


def get_transforms(train=True):  # Trả về các phép biến đổi ảnh; train=True cho tập train, False cho validation/test
    if train:  # Với tập train: thêm augmentation (biến đổi ngẫu nhiên) để mô hình đỡ học thuộc lòng (overfit)
        return Compose([  # Compose: nối nhiều phép biến đổi, chạy lần lượt từ trên xuống
            RandomCrop(32, padding=4),  # Đệm 4 pixel quanh ảnh (thành 40x40) rồi cắt ngẫu nhiên một vùng 32x32 -> ảnh bị dịch chuyển nhẹ
            RandomHorizontalFlip(),  # Lật ngang ảnh với xác suất 50% (con mèo quay trái hay phải vẫn là con mèo)
            Normalize(CIFAR_MEAN, CIFAR_STD),  # Chuẩn hóa: (giá trị - mean) / std -> dữ liệu có trung bình ~0, độ lệch ~1, giúp học nhanh hơn
        ])
    return Normalize(CIFAR_MEAN, CIFAR_STD)  # Với validation/test: chỉ chuẩn hóa, không biến đổi ngẫu nhiên để kết quả ổn định


def load_class_names(root="data"):  # Đọc tên 10 lớp (airplane, automobile, ...) từ file batches.meta
    with open(os.path.join(root, "cifar-10-batches-py", "batches.meta"), 'rb') as fo:  # Mở file ở chế độ đọc nhị phân ('rb')
        meta = pickle.load(fo, encoding='bytes')  # Giải nén nội dung; encoding='bytes' vì file được tạo bằng Python 2
    return [name.decode() for name in meta[b'label_names']]  # Tên lớp đang ở dạng bytes (b'cat'), decode() đổi sang chuỗi 'cat'


class CIFARDataset(Dataset):  # Bộ dữ liệu CIFAR-10, kế thừa Dataset để dùng được với DataLoader
    def __init__(self, root="data", train=True, transform=None):  # Hàm khởi tạo: root = thư mục dữ liệu, train = lấy tập train hay test, transform = phép biến đổi ảnh
        data_path = os.path.join(root, "cifar-10-batches-py")  # Ghép đường dẫn tới thư mục chứa các file batch
        if train:  # Nếu lấy tập train
            data_files = [os.path.join(data_path, "data_batch_{}".format(i)) for i in range(1, 6)]  # 5 file data_batch_1 ... data_batch_5, mỗi file 10.000 ảnh
        else:  # Nếu lấy tập test
            data_files = [os.path.join(data_path, "test_batch")]  # 1 file test_batch gồm 10.000 ảnh

        images = []  # Danh sách tạm chứa mảng ảnh của từng file batch
        labels = []  # Danh sách nhãn (số nguyên 0-9) của tất cả ảnh
        for data_file in data_files:  # Duyệt qua từng file batch
            with open(data_file, 'rb') as fo:  # Mở file ở chế độ đọc nhị phân
                batch = pickle.load(fo, encoding='bytes')  # batch là một dict, key ở dạng bytes: b'data', b'labels', ...
                images.append(batch[b'data'])  # b'data' là mảng uint8 kích thước (10000, 3072): mỗi hàng là một ảnh duỗi phẳng
                labels.extend(batch[b'labels'])  # b'labels' là list 10.000 nhãn; extend nối từng phần tử vào danh sách
        # Giữ dạng uint8 (N, 3, 32, 32) để tiết kiệm RAM, chỉ đổi sang float khi lấy từng ảnh
        self.images = np.concatenate(images).reshape(-1, 3, 32, 32)  # Ghép các batch thành (N, 3072) rồi đổi hình thành (N, kênh, cao, rộng); -1 = tự tính N
        self.labels = labels  # Lưu danh sách nhãn
        self.transform = transform  # Lưu phép biến đổi để dùng trong __getitem__

    def __len__(self):  # DataLoader gọi hàm này để biết bộ dữ liệu có bao nhiêu mẫu
        return len(self.labels)  # Số mẫu = số nhãn (50.000 với train, 10.000 với test)

    def __getitem__(self, idx):  # DataLoader gọi hàm này để lấy mẫu thứ idx
        image = torch.from_numpy(self.images[idx]).float() / 255.  # Lấy ảnh uint8 (0-255), đổi sang tensor float rồi chia 255 -> giá trị trong [0, 1]
        if self.transform:  # Nếu có phép biến đổi
            image = self.transform(image)  # Áp dụng augmentation/chuẩn hóa
        return image, self.labels[idx]  # Trả về cặp (ảnh tensor (3, 32, 32), nhãn số nguyên)


if __name__ == '__main__':  # Chỉ chạy khối này khi chạy trực tiếp file (python dataset.py), không chạy khi file được import
    dataset = CIFARDataset(root="./data", train=True, transform=get_transforms(train=True))  # Tạo bộ dữ liệu train có augmentation
    print(len(dataset), load_class_names("./data"))  # In số mẫu và tên các lớp để kiểm tra
    train_dataloader = DataLoader(  # Tạo DataLoader để lấy dữ liệu theo batch
        dataset=dataset,  # Bộ dữ liệu nguồn
        batch_size=4,  # Mỗi batch gồm 4 ảnh
        shuffle=True  # Xáo trộn thứ tự ảnh
    )
    for images, labels in train_dataloader:  # Lấy từng batch: images có kích thước (4, 3, 32, 32), labels có kích thước (4,)
        print(images.shape)  # In kích thước batch ảnh
        print(labels)  # In nhãn của batch
        break  # Chỉ cần kiểm tra batch đầu tiên rồi dừng

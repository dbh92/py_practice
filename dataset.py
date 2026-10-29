import os  # Thư viện làm việc với file/thư mục
import torch  # Thư viện PyTorch chính
from PIL import Image  # Thư viện mở file ảnh
from torch.utils.data import Dataset, DataLoader  # Dataset: định nghĩa cách lấy từng ảnh; DataLoader: gom ảnh thành batch
from torchvision import transforms  # Các phép biến đổi ảnh (resize, lật, chuẩn hóa, ...)
import config  # Các thiết lập của dự án (config.py)


def get_class_names(split_dir=config.SPLIT_DIR):  # Đọc tên các lớp từ tên thư mục con của data/split/train
    train_dir = os.path.join(split_dir, "train")  # Đường dẫn tới tập train
    return sorted(name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name)))  # Sắp xếp theo tên để thứ tự lớp luôn cố định (vd: ['NG', 'OK'])


def build_transforms(train, image_size=config.IMAGE_SIZE):  # Tạo chuỗi biến đổi ảnh; train=True thêm augmentation
    steps = [transforms.Resize((image_size, image_size))]  # Bước 1: thu nhỏ/phóng to ảnh về kích thước cố định
    if train:  # Chỉ tập train mới được biến đổi ngẫu nhiên; val/test phải giữ nguyên để đánh giá công bằng
        if config.HORIZONTAL_FLIP:  # Nếu bật lật ngang
            steps.append(transforms.RandomHorizontalFlip())  # Lật ngang với xác suất 50%
        if config.VERTICAL_FLIP:  # Nếu bật lật dọc
            steps.append(transforms.RandomVerticalFlip())  # Lật dọc với xác suất 50%
        if config.ROTATION_DEGREES:  # Nếu góc xoay khác 0
            steps.append(transforms.RandomRotation(config.ROTATION_DEGREES))  # Xoay ngẫu nhiên (góc trống được tô đen)
        if config.BRIGHTNESS or config.CONTRAST:  # Nếu bật thay đổi độ sáng/tương phản
            steps.append(transforms.ColorJitter(brightness=config.BRIGHTNESS, contrast=config.CONTRAST))  # Đổi ngẫu nhiên độ sáng, tương phản
    steps.append(transforms.ToTensor())  # Đổi ảnh PIL thành tensor (3, H, W) với giá trị trong [0, 1]
    steps.append(transforms.Normalize(config.MEAN, config.STD))  # Chuẩn hóa: (giá trị - mean) / std
    return transforms.Compose(steps)  # Nối tất cả các bước thành một phép biến đổi


class ImageClassificationDataset(Dataset):  # Đọc ảnh theo cấu trúc thư mục: <root>/<tên lớp>/<ảnh>
    def __init__(self, root, class_names, transform=None):  # root: vd data/split/train; class_names: danh sách lớp cố định
        self.samples = []  # Danh sách các cặp (đường dẫn ảnh, nhãn số); chỉ lưu đường dẫn, ảnh được mở khi cần để tiết kiệm RAM
        for label, class_name in enumerate(class_names):  # enumerate đánh số lớp: 0 = class_names[0], 1 = class_names[1], ...
            class_dir = os.path.join(root, class_name)  # Thư mục của lớp, vd: data/split/train/NG
            if not os.path.isdir(class_dir):  # Tập này không có thư mục của lớp đó (vd: quá ít ảnh nên bị trống)
                continue  # Bỏ qua
            for name in sorted(os.listdir(class_dir)):  # Duyệt các file trong thư mục lớp
                if name.lower().endswith(config.IMAGE_EXTENSIONS):  # Chỉ lấy file ảnh
                    self.samples.append((os.path.join(class_dir, name), label))  # Lưu cặp (đường dẫn, nhãn)
        self.transform = transform  # Lưu phép biến đổi

    def __len__(self):  # Số ảnh trong bộ dữ liệu
        return len(self.samples)  # Bằng số cặp đã lưu

    def __getitem__(self, idx):  # Lấy ảnh thứ idx
        path, label = self.samples[idx]  # Lấy đường dẫn và nhãn
        image = Image.open(path).convert("RGB")  # Mở ảnh và ép về 3 kênh màu (ảnh xám từ camera công nghiệp cũng thành RGB)
        if self.transform:  # Nếu có phép biến đổi
            image = self.transform(image)  # Resize, augmentation, chuẩn hóa
        return image, label  # Trả về (tensor ảnh, nhãn số)


def make_dataloader(split, class_names, train, image_size=config.IMAGE_SIZE):  # Tạo DataLoader cho một tập: "train", "val" hoặc "test"
    dataset = ImageClassificationDataset(
        root=os.path.join(config.SPLIT_DIR, split),  # Thư mục của tập, vd: data/split/val
        class_names=class_names,  # Dùng chung danh sách lớp để nhãn số giống nhau ở mọi tập
        transform=build_transforms(train, image_size),  # train=True mới có augmentation
    )
    return DataLoader(
        dataset=dataset,  # Bộ dữ liệu nguồn
        batch_size=config.BATCH_SIZE,  # Số ảnh mỗi batch
        shuffle=train,  # Chỉ xáo trộn tập train
        num_workers=config.NUM_WORKERS,  # Số tiến trình phụ đọc ảnh song song
        persistent_workers=config.NUM_WORKERS > 0,  # Giữ tiến trình phụ qua các epoch, đỡ tốn thời gian tạo lại trên Windows
        pin_memory=torch.cuda.is_available(),  # Có GPU thì chép dữ liệu sang GPU nhanh hơn
    )


if __name__ == '__main__':  # Chạy trực tiếp file để kiểm tra dữ liệu đã chia đúng chưa
    class_names = get_class_names()  # Đọc tên lớp
    print("Classes: {}".format(class_names))  # In danh sách lớp, vd: ['NG', 'OK']
    for split in ("train", "val", "test"):  # Với từng tập
        loader = make_dataloader(split, class_names, train=(split == "train"))  # Tạo DataLoader
        counts = [sum(1 for _, label in loader.dataset.samples if label == i) for i in range(len(class_names))]  # Đếm số ảnh mỗi lớp
        print("{:<5} {} images, per class: {}".format(split, len(loader.dataset), dict(zip(class_names, counts))))  # In thống kê
    images, labels = next(iter(loader))  # Lấy thử một batch của tập cuối (test)
    print("Batch shape: {}, labels: {}".format(tuple(images.shape), labels.tolist()))  # Kích thước phải là (batch, 3, IMAGE_SIZE, IMAGE_SIZE)

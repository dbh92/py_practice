import argparse  # Thư viện đọc tham số dòng lệnh
import os  # Thư viện làm việc với file/thư mục
import torch  # Thư viện PyTorch chính
import matplotlib.pyplot as plt  # Thư viện vẽ hình, dùng để hiện ảnh khi có cờ --show
from PIL import Image  # Thư viện mở file ảnh (jpg, png, ...)
from torchvision.transforms import Compose, Resize, ToTensor  # Các phép biến đổi ảnh
from dataset import get_transforms, load_class_names  # Dùng lại phép chuẩn hóa và tên lớp từ dataset.py
from models import MODELS  # Bảng tra các mô hình trong models.py

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")  # Các đuôi file được coi là ảnh khi đưa vào cả thư mục


def get_args():  # Định nghĩa và đọc các tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Predict the class of any image with a trained CIFAR-10 model")  # Tạo bộ phân tích tham số
    parser.add_argument("paths", nargs="+", help="image files or folders")  # Một hoặc nhiều đường dẫn ảnh/thư mục (bắt buộc)
    parser.add_argument("--model", choices=MODELS.keys(), default="cnn")  # Mô hình nào: phải khớp với mô hình đã train
    parser.add_argument("--checkpoint", default=None, help="default: <checkpoint-dir>/<model>_best.pt")  # Đường dẫn file trọng số (không ghi thì tự tìm)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")  # Thư mục chứa checkpoint
    parser.add_argument("--data-root", default="./data")  # Thư mục dữ liệu (chỉ để đọc tên 10 lớp)
    parser.add_argument("--top-k", type=int, default=3)  # In ra bao nhiêu lớp có xác suất cao nhất
    parser.add_argument("--show", action="store_true")  # Có cờ này thì hiện ảnh gốc và ảnh 32x32 mà mô hình thực sự "nhìn thấy"
    return parser.parse_args()  # Trả về đối tượng args


def collect_image_paths(paths):  # Gom danh sách file ảnh: file thì giữ nguyên, thư mục thì lấy mọi ảnh bên trong
    image_paths = []  # Danh sách kết quả
    for path in paths:  # Duyệt từng đường dẫn người dùng đưa vào
        if os.path.isdir(path):  # Nếu là thư mục
            for name in sorted(os.listdir(path)):  # Duyệt các file trong thư mục theo thứ tự tên
                if name.lower().endswith(IMAGE_EXTENSIONS):  # Chỉ lấy file có đuôi ảnh (không phân biệt hoa/thường)
                    image_paths.append(os.path.join(path, name))  # Thêm đường dẫn đầy đủ
        else:  # Nếu là file
            image_paths.append(path)  # Thêm luôn
    return image_paths  # Trả về danh sách


if __name__ == '__main__':  # Chỉ chạy khi chạy trực tiếp file
    args = get_args()  # Đọc tham số dòng lệnh
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Dùng GPU nếu có, không thì CPU

    checkpoint_path = args.checkpoint or os.path.join(args.checkpoint_dir, "{}_best.pt".format(args.model))  # Không chỉ định thì dùng checkpoint tốt nhất, vd: checkpoints/cnn_best.pt
    if not os.path.isfile(checkpoint_path):  # Nếu chưa có file trọng số (chưa train mô hình này)
        print("Checkpoint not found: {}. Train it first: python train_neuralnetwork.py --model {}".format(
            checkpoint_path, args.model))  # Báo cho người dùng biết cần làm gì (in tiếng Anh vì console Windows có thể không in được dấu tiếng Việt)
        raise SystemExit(1)  # Dừng chương trình

    model = MODELS[args.model](num_classes=10).to(device)  # Tạo mô hình rỗng cùng kiến trúc với lúc train
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))  # Nạp trọng số đã học vào mô hình
    model.eval()  # Chế độ đánh giá: tắt Dropout, BatchNorm dùng thống kê đã học (quên dòng này sẽ cho kết quả sai)
    class_names = load_class_names(args.data_root)  # Tên 10 lớp: airplane, automobile, bird, ...

    # Xử lý ảnh giống hệt lúc train: ảnh 32x32, RGB, giá trị [0, 1], rồi chuẩn hóa theo mean/std của CIFAR-10
    transform = Compose([  # Nối các phép biến đổi
        Resize((32, 32)),  # Thu nhỏ ảnh bất kỳ về 32x32 (kích thước mô hình được train)
        ToTensor(),  # Đổi ảnh PIL thành tensor (3, 32, 32) với giá trị trong [0, 1]
        get_transforms(train=False),  # Chuẩn hóa (giá trị - mean) / std, đúng như tập test lúc train
    ])

    for image_path in collect_image_paths(args.paths):  # Duyệt từng ảnh cần dự đoán
        try:  # Thử mở ảnh
            image = Image.open(image_path).convert("RGB")  # Mở ảnh và ép về 3 kênh màu (ảnh xám hoặc PNG trong suốt cũng thành RGB)
        except OSError as error:  # File không tồn tại hoặc không phải ảnh
            print("Skipping {}: {}".format(image_path, error))  # Báo lỗi (tiếng Anh, lý do như trên)
            continue  # Sang ảnh kế tiếp

        input_tensor = transform(image).unsqueeze(0).to(device)  # (3, 32, 32) -> (1, 3, 32, 32): mô hình luôn nhận một batch, ở đây batch gồm 1 ảnh
        with torch.no_grad():  # Không cần tính gradient khi dự đoán
            probabilities = torch.softmax(model(input_tensor), dim=1)[0]  # softmax đổi 10 điểm số (logits) thành 10 xác suất có tổng = 1; [0] lấy ảnh đầu tiên
        top_probs, top_indices = probabilities.topk(min(args.top_k, len(class_names)))  # Lấy k xác suất lớn nhất và chỉ số lớp tương ứng

        print(image_path)  # In tên file ảnh
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):  # Duyệt từng cặp (xác suất, chỉ số lớp)
            print("  {:<12} {:6.2%}".format(class_names[idx], prob))  # In tên lớp (căn trái 12 ký tự) và xác suất dạng phần trăm

        if args.show:  # Nếu người dùng muốn xem ảnh
            fig, (ax_original, ax_small) = plt.subplots(1, 2, figsize=(8, 4))  # Tạo cửa sổ có 2 ô hình cạnh nhau
            ax_original.imshow(image)  # Ô trái: ảnh gốc
            ax_original.set_title("Ảnh gốc {}x{}".format(*image.size))  # Tiêu đề kèm kích thước gốc (rộng x cao)
            ax_small.imshow(image.resize((32, 32)))  # Ô phải: ảnh 32x32 mà mô hình thực sự nhận vào
            ax_small.set_title("{} ({:.1%})".format(class_names[top_indices[0]], top_probs[0].item()))  # Tiêu đề: lớp được đoán và xác suất
            for ax in (ax_original, ax_small):  # Với cả 2 ô hình
                ax.axis("off")  # Ẩn trục tọa độ cho gọn
            fig.suptitle(os.path.basename(image_path))  # Tiêu đề chung: tên file
            plt.show()  # Hiện cửa sổ; đóng cửa sổ thì chương trình chạy tiếp sang ảnh sau

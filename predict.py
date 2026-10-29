import argparse  # Thư viện đọc tham số dòng lệnh
import math  # Thư viện toán học, dùng hàm ceil (làm tròn lên) để tính số trang, số hàng
import os  # Thư viện làm việc với file/thư mục
import torch  # Thư viện PyTorch chính
import matplotlib.pyplot as plt  # Thư viện vẽ hình, dùng để hiện ảnh khi có cờ --show
from PIL import Image  # Thư viện mở file ảnh
import config  # Các thiết lập của dự án (config.py)
from dataset import build_transforms  # Dùng lại đúng chuỗi xử lý ảnh của tập val/test
from model import load_model_from_checkpoint  # Nạp mô hình đã train
from evaluate import decide_predictions  # Quy tắc ra kết luận (argmax hoặc ngưỡng NG), giống hệt evaluate.py

IMAGES_PER_PAGE = 20  # Số ảnh mỗi trang khi --show/--save (tối đa 4 hàng x 5 cột)


def collect_image_paths(paths):  # Gom danh sách file ảnh: file thì giữ nguyên, thư mục thì lấy mọi ảnh bên trong
    image_paths = []  # Danh sách kết quả
    for path in paths:  # Duyệt từng đường dẫn người dùng đưa vào
        if os.path.isdir(path):  # Nếu là thư mục
            for name in sorted(os.listdir(path)):  # Duyệt các file theo thứ tự tên
                if name.lower().endswith(config.IMAGE_EXTENSIONS):  # Chỉ lấy file ảnh
                    image_paths.append(os.path.join(path, name))  # Thêm đường dẫn đầy đủ
        else:  # Nếu là file
            image_paths.append(path)  # Thêm luôn
    return image_paths  # Trả về danh sách


def true_label_from_path(path, class_names):  # Đoán nhãn thật từ tên thư mục chứa ảnh, vd data/split/test/NG/abc.jpg -> "NG"
    folder = os.path.basename(os.path.dirname(os.path.abspath(path)))  # Tên thư mục cha của file ảnh
    return folder if folder in class_names else None  # Thư mục không phải tên lớp (vd ảnh mới chưa biết) thì None


def show_results(results, show, save_dir):  # Vẽ ảnh kèm kết luận thành lưới, mỗi trang IMAGES_PER_PAGE ảnh
    num_pages = math.ceil(len(results) / IMAGES_PER_PAGE)  # Số trang cần vẽ
    for page in range(num_pages):  # Vẽ từng trang
        chunk = results[page * IMAGES_PER_PAGE:(page + 1) * IMAGES_PER_PAGE]  # Các ảnh thuộc trang này
        cols = min(5, len(chunk))  # Tối đa 5 cột
        rows = math.ceil(len(chunk) / cols)  # Số hàng cần thiết
        fig, axes = plt.subplots(rows, cols, figsize=(max(3 * cols, 7), 3.4 * rows + 0.4), squeeze=False)  # Tạo lưới ô hình; rộng tối thiểu 7 để tiêu đề trang không bị cắt khi chỉ có 1-2 ảnh; squeeze=False để axes luôn là bảng 2 chiều
        for ax in axes.flat:  # Với mọi ô
            ax.axis("off")  # Ẩn trục tọa độ (ô thừa ở cuối trang sẽ để trống)
        for ax, result in zip(axes.flat, chunk):  # Mỗi ảnh một ô
            ax.imshow(result["thumbnail"])  # Hiện ảnh thu nhỏ
            if result["true"] is None:  # Không biết nhãn thật
                color = "black"  # Màu đen
            elif result["true"] == result["predicted"]:  # Đoán đúng
                color = "green"  # Màu xanh
            else:  # Đoán sai
                color = "red"  # Màu đỏ
            title = "{} {:.1%}".format(result["predicted"], result["confidence"])  # Dòng 1: kết luận và độ tự tin
            if color == "red":  # Đoán sai thì ghi thêm nhãn thật
                title += " (true {})".format(result["true"])  # vd: "OK 94.1% (true NG)"
            ax.set_title(title + "\n" + result["name"][-28:], color=color, fontsize=9)  # Dòng 2: tên file (cắt bớt nếu quá dài)
        fig.suptitle("Page {}/{} - green: correct, red: wrong, black: unknown true label".format(page + 1, num_pages))  # Tiêu đề trang
        fig.tight_layout()  # Tự căn chỉnh khoảng cách cho khỏi chồng chữ
        if save_dir:  # Nếu muốn lưu ra file
            os.makedirs(save_dir, exist_ok=True)  # Tạo thư mục nếu chưa có
            figure_path = os.path.join(save_dir, "predictions_{:02d}.png".format(page + 1))  # vd: predictions_01.png
            fig.savefig(figure_path, dpi=100)  # Lưu hình
            print("Saved {}".format(figure_path))  # Báo đã lưu
        if show:  # Nếu muốn hiện cửa sổ
            plt.show()  # Hiện trang này; đóng cửa sổ thì sang trang tiếp theo
        plt.close(fig)  # Đóng hình để giải phóng bộ nhớ


if __name__ == '__main__':  # Chỉ chạy khi chạy trực tiếp file
    parser = argparse.ArgumentParser(description="Predict OK/NG for any image")  # Tạo bộ phân tích tham số
    parser.add_argument("paths", nargs="+", help="image files or folders")  # Một hoặc nhiều ảnh/thư mục (bắt buộc)
    parser.add_argument("--checkpoint", default=os.path.join(config.OUTPUT_DIR, "best.pt"))  # File mô hình
    parser.add_argument("--ng-threshold", type=float, default=config.NG_THRESHOLD)  # Ngưỡng NG, mặc định lấy từ config.py
    parser.add_argument("--show", action="store_true", help="show images with predictions, 20 per window")  # Hiện ảnh kèm kết luận dạng lưới
    parser.add_argument("--save", metavar="FOLDER", help="save the image grids as PNG files in FOLDER")  # Lưu lưới ảnh ra file
    parser.add_argument("--only-wrong", action="store_true", help="show/save only wrong predictions (needs OK/NG folder names)")  # Chỉ hiện ảnh đoán sai
    args = parser.parse_args()  # Đọc tham số

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Dùng GPU nếu có
    if not os.path.isfile(args.checkpoint):  # Chưa có mô hình
        print("Checkpoint not found: {}. Run train.py first.".format(args.checkpoint))  # Hướng dẫn (in tiếng Anh vì console Windows có thể không in được dấu tiếng Việt)
        raise SystemExit(1)  # Dừng chương trình
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)  # Nạp mô hình
    class_names = checkpoint["class_names"]  # Danh sách lớp lúc train
    transform = build_transforms(train=False, image_size=checkpoint["image_size"])  # Resize + chuẩn hóa giống hệt lúc đánh giá

    results = []  # Kết quả của từng ảnh, dùng để vẽ khi có --show/--save
    for image_path in collect_image_paths(args.paths):  # Duyệt từng ảnh
        try:  # Thử mở ảnh
            image = Image.open(image_path).convert("RGB")  # Mở ảnh và ép về 3 kênh màu
        except OSError as error:  # File không tồn tại hoặc không phải ảnh
            print("Skipping {}: {}".format(image_path, error))  # Báo lỗi rồi chuyển sang ảnh tiếp theo
            continue  # Sang ảnh kế tiếp

        input_tensor = transform(image).unsqueeze(0).to(device)  # (3, H, W) -> (1, 3, H, W): batch gồm 1 ảnh
        with torch.no_grad():  # Không cần gradient khi dự đoán
            probs = torch.softmax(model(input_tensor), dim=1)[0].tolist()  # Xác suất từng lớp của ảnh này
        prediction = decide_predictions([probs], class_names, args.ng_threshold)[0]  # Kết luận lớp (chỉ số)
        true_label = true_label_from_path(image_path, class_names)  # Nhãn thật nếu ảnh nằm trong thư mục OK/NG
        wrong = true_label is not None and true_label != class_names[prediction]  # Có biết nhãn thật và đoán sai không
        prob_text = "  ".join("{} {:6.2%}".format(name, p) for name, p in zip(class_names, probs))  # Chuỗi xác suất, vd "NG  3.10%  OK 96.90%"
        print("{:<4} | {} | {}{}".format(class_names[prediction], prob_text, image_path,
                                         "  <- WRONG (true {})".format(true_label) if wrong else ""))  # In kết luận, xác suất, tên file; đánh dấu nếu sai

        if (args.show or args.save) and (wrong or not args.only_wrong):  # Chỉ giữ ảnh cần vẽ
            thumbnail = image.copy()  # Sao chép ảnh để thu nhỏ (không làm hỏng ảnh gốc)
            thumbnail.thumbnail((256, 256))  # Thu nhỏ giữ tỉ lệ, cạnh dài tối đa 256 pixel -> đỡ tốn RAM khi có nhiều ảnh
            results.append({  # Lưu thông tin để vẽ
                "thumbnail": thumbnail,  # Ảnh thu nhỏ
                "name": os.path.basename(image_path),  # Tên file
                "predicted": class_names[prediction],  # Lớp được đoán
                "confidence": probs[prediction],  # Xác suất của lớp được đoán
                "true": true_label,  # Nhãn thật (hoặc None)
            })

    if args.show or args.save:  # Có yêu cầu vẽ
        if results:  # Có ảnh để vẽ
            show_results(results, args.show, args.save)  # Vẽ theo từng trang
        else:  # Không có ảnh nào (vd --only-wrong mà không có ảnh sai)
            print("No images to show.")  # Thông báo

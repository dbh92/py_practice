import argparse  # Thư viện đọc tham số dòng lệnh
import csv  # Thư viện ghi file CSV (mở được bằng Excel)
import os  # Thư viện làm việc với file/thư mục
import torch  # Thư viện PyTorch chính
import matplotlib  # Thư viện vẽ hình
import matplotlib.pyplot as plt  # Hàm vẽ hình
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score  # Các chỉ số đánh giá
import config  # Các thiết lập của dự án (config.py)
from dataset import make_dataloader  # Tạo DataLoader cho một tập
from model import load_model_from_checkpoint  # Nạp mô hình đã train


def run_inference(model, dataloader, device, criterion=None):  # Cho mô hình dự đoán toàn bộ một tập; trả về (loss trung bình, nhãn thật, xác suất)
    model.eval()  # Chế độ đánh giá: tắt Dropout, BatchNorm dùng thống kê đã học
    total_loss = 0.  # Cộng dồn loss (chỉ tính khi có criterion)
    all_labels = []  # Nhãn thật của mọi ảnh
    all_probabilities = []  # Xác suất từng lớp của mọi ảnh, vd [[0.9, 0.1], [0.2, 0.8], ...]
    with torch.no_grad():  # Không tính gradient -> nhanh và tốn ít bộ nhớ
        for images, labels in dataloader:  # Duyệt từng batch
            outputs = model(images.to(device))  # Logits (batch, số lớp)
            if criterion is not None:  # Nếu cần tính loss (lúc train, để theo dõi val loss)
                total_loss += criterion(outputs, labels.to(device)).item() * images.size(0)  # Loss trung bình của batch x số ảnh = tổng loss của batch
            all_probabilities.extend(torch.softmax(outputs, dim=1).cpu().tolist())  # softmax đổi logits thành xác suất có tổng = 1
            all_labels.extend(labels.tolist())  # Thêm nhãn thật
    average_loss = total_loss / len(dataloader.dataset) if criterion is not None else None  # Loss trung bình trên mỗi ảnh
    return average_loss, all_labels, all_probabilities  # Trả về kết quả


def decide_predictions(probabilities, class_names, ng_threshold=None):  # Từ xác suất ra kết luận lớp cho từng ảnh
    ng_index = class_names.index(config.NG_CLASS) if config.NG_CLASS in class_names else None  # Vị trí lớp NG trong danh sách lớp (None nếu không có)
    predictions = []  # Kết luận của từng ảnh
    for probs in probabilities:  # Với xác suất của từng ảnh
        prediction = probs.index(max(probs))  # Mặc định: chọn lớp có xác suất cao nhất
        if ng_threshold is not None and ng_index is not None:  # Nếu dùng ngưỡng NG riêng
            if probs[ng_index] >= ng_threshold:  # Xác suất NG đạt ngưỡng
                prediction = ng_index  # -> kết luận NG
            elif prediction == ng_index:  # NG cao nhất nhưng chưa đạt ngưỡng
                others = [p if i != ng_index else -1. for i, p in enumerate(probs)]  # Loại NG ra khỏi lựa chọn
                prediction = others.index(max(others))  # -> chọn lớp cao nhất còn lại
        predictions.append(prediction)  # Lưu kết luận
    return predictions  # Trả về danh sách kết luận (chỉ số lớp)


def compute_metrics(labels, predictions, class_names):  # Tính các chỉ số chính: accuracy, F1, recall của NG
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)  # Tỉ lệ đoán đúng trên tổng số ảnh
    macro_f1 = f1_score(labels, predictions, labels=list(range(len(class_names))), average="macro", zero_division=0)  # F1 trung bình các lớp: lớp ít ảnh (NG) quan trọng ngang lớp nhiều ảnh (OK)
    ng_recall = None  # Recall của NG = số NG bắt được / tổng số NG thật
    if config.NG_CLASS in class_names:  # Chỉ tính khi có lớp NG
        ng_index = class_names.index(config.NG_CLASS)  # Vị trí lớp NG
        ng_recall = recall_score(labels, predictions, labels=[ng_index], average="macro", zero_division=0)  # Recall riêng của lớp NG
    return {"accuracy": accuracy, "macro_f1": macro_f1, "ng_recall": ng_recall}  # Trả về dạng dict


if __name__ == '__main__':  # Chạy trực tiếp file để đánh giá mô hình đã train
    matplotlib.use("Agg")  # Chỉ lưu hình ra file, không mở cửa sổ (đặt ở đây để predict.py import file này vẫn mở được cửa sổ --show)
    parser =argparse.ArgumentParser(description="Evaluate a trained model on the val or test split")  # Tạo bộ phân tích tham số
    parser.add_argument("--checkpoint", default=os.path.join(config.OUTPUT_DIR, "best.pt"))  # File mô hình cần đánh giá
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")  # Đánh giá trên tập nào (mặc định test)
    parser.add_argument("--ng-threshold", type=float, default=config.NG_THRESHOLD)  # Ngưỡng NG, mặc định lấy từ config.py
    args = parser.parse_args()  # Đọc tham số

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Dùng GPU nếu có
    if not os.path.isfile(args.checkpoint):  # Chưa có mô hình
        print("Checkpoint not found: {}. Run train.py first.".format(args.checkpoint))  # Hướng dẫn
        raise SystemExit(1)  # Dừng chương trình
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)  # Nạp mô hình
    class_names = checkpoint["class_names"]  # Danh sách lớp lúc train (thứ tự phải giữ nguyên)
    dataloader = make_dataloader(args.split, class_names, train=False, image_size=checkpoint["image_size"])  # DataLoader của tập cần đánh giá, cùng kích thước ảnh lúc train
    _, labels, probabilities = run_inference(model, dataloader, device)  # Dự đoán toàn bộ tập
    predictions = decide_predictions(probabilities, class_names, args.ng_threshold)  # Ra kết luận từng ảnh

    print("Model: {} (epoch {}) | split: {} ({} images) | NG threshold: {}".format(
        checkpoint["model_name"], checkpoint["epoch"], args.split, len(labels),
        args.ng_threshold if args.ng_threshold is not None else "argmax"))  # In thông tin đánh giá
    print(classification_report(labels, predictions, labels=list(range(len(class_names))), target_names=class_names,
                                digits=4, zero_division=0))  # Bảng precision/recall/F1 từng lớp

    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))  # Ma trận nhầm lẫn: hàng = lớp thật, cột = lớp đoán
    print("Confusion matrix (rows = true, columns = predicted):")  # Tiêu đề
    print(" " * 14 + "".join("{:>10}".format(name) for name in class_names))  # Dòng tên cột
    for name, row in zip(class_names, matrix):  # Mỗi hàng là một lớp thật
        print("{:<14}".format(name) + "".join("{:>10}".format(value) for value in row))  # In số ảnh từng ô

    if config.NG_CLASS in class_names:  # Tóm tắt theo góc nhìn QC
        ng_index = class_names.index(config.NG_CLASS)  # Vị trí lớp NG
        total_ng = int(matrix[ng_index].sum())  # Tổng số ảnh NG thật
        missed_ng = total_ng - int(matrix[ng_index][ng_index])  # NG thật nhưng bị đoán thành lớp khác = lọt lỗi (nguy hiểm nhất)
        total_other = int(matrix.sum()) - total_ng  # Tổng số ảnh không phải NG
        false_alarms = int(matrix[:, ng_index].sum()) - int(matrix[ng_index][ng_index])  # Không phải NG nhưng bị đoán NG = báo nhầm
        print("\nQC summary: missed NG (escape) {}/{} | false alarm {}/{}".format(
            missed_ng, total_ng, false_alarms, total_other))  # In tóm tắt

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)  # Tạo thư mục outputs nếu chưa có
    figure_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix_{}.png".format(args.split))  # Đường dẫn file hình
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(cmap="Blues", values_format="d")  # Vẽ ma trận nhầm lẫn (ô càng đậm càng nhiều ảnh)
    plt.title("{} - {} split".format(checkpoint["model_name"], args.split))  # Tiêu đề hình
    plt.savefig(figure_path, dpi=120, bbox_inches="tight")  # Lưu hình
    plt.close()  # Đóng hình để giải phóng bộ nhớ

    csv_path = os.path.join(config.OUTPUT_DIR, "misclassified_{}.csv".format(args.split))  # File liệt kê ảnh bị đoán sai
    wrong = [(path, label, prediction, probs) for (path, label), prediction, probs
             in zip(dataloader.dataset.samples, predictions, probabilities) if prediction != label]  # Lọc các ảnh đoán sai (thứ tự samples khớp vì DataLoader không xáo trộn)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig để Excel đọc đúng tên file có dấu tiếng Việt
        writer = csv.writer(f)  # Bộ ghi CSV
        writer.writerow(["image", "true", "predicted"] + ["prob_" + name for name in class_names])  # Dòng tiêu đề
        for path, label, prediction, probs in wrong:  # Mỗi ảnh đoán sai một dòng
            writer.writerow([path, class_names[label], class_names[prediction]] + ["{:.4f}".format(p) for p in probs])  # Ghi đường dẫn, nhãn thật, nhãn đoán, xác suất
    print("\n{} misclassified images -> {}".format(len(wrong), csv_path))  # Báo số ảnh sai
    for path, label, prediction, _ in wrong[:10]:  # In tối đa 10 ảnh sai đầu tiên để xem nhanh
        print("  {} (true {}, predicted {})".format(path, class_names[label], class_names[prediction]))  # Đường dẫn và nhãn
    print("Confusion matrix figure -> {}".format(figure_path))  # Báo nơi lưu hình

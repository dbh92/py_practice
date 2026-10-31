# Gửi ảnh lên service để thử API, in kết quả giống predict.py.
# Thêm --compare để kiểm chứng: kết quả của service phải trùng với PyTorch (cùng cách tính như predict.py).
# --compare cần PyTorch, nên chỉ chạy trên máy train (trong .venv của dự án); không có --compare thì chỉ cần thư viện requests.
# Chạy (service phải đang chạy): python service/test_client.py data/split/test/NG data/split/test/OK --compare

import argparse  # Thư viện đọc tham số dòng lệnh
import os  # Thư viện làm việc với file/thư mục
import sys  # Thư viện hệ thống, dùng để thêm đường dẫn import
import time  # Thư viện đo thời gian
import requests  # Thư viện gửi yêu cầu HTTP

SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))  # Thư mục service/
PROJECT_DIR = os.path.dirname(SERVICE_DIR)  # Thư mục gốc dự án
sys.path.insert(0, os.path.join(PROJECT_DIR, "export_onnx"))  # Để dùng lại collect_image_paths trong predict_onnx.py
from predict_onnx import collect_image_paths  # Gom danh sách ảnh từ file/thư mục
MAX_PROB_DIFF = 1e-4  # Sai lệch xác suất tối đa cho phép giữa service và PyTorch (0.01%)


def load_pytorch_reference(checkpoint_path):  # Nạp mô hình PyTorch để làm chuẩn so sánh; trả về hàm tính xác suất cho 1 ảnh
    sys.path.insert(0, PROJECT_DIR)  # Để import config.py, model.py... ở thư mục gốc
    import torch  # Chỉ import khi có --compare, để máy không có PyTorch vẫn chạy được phần gọi API
    from PIL import Image  # Thư viện mở file ảnh
    from dataset import build_transforms  # Cùng chuỗi xử lý ảnh với predict.py
    from model import load_model_from_checkpoint  # Nạp mô hình đã train
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, torch.device("cpu"))  # Nạp lên CPU, đã ở chế độ eval
    transform = build_transforms(train=False, image_size=checkpoint["image_size"])  # Resize + chuẩn hóa giống predict.py

    def pytorch_probabilities(image_path):  # Xác suất từng lớp của 1 ảnh, tính đúng như predict.py
        image = Image.open(image_path).convert("RGB")  # Mở ảnh, ép về 3 kênh màu
        with torch.no_grad():  # Không cần gradient
            return torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0].tolist()  # Ảnh -> tensor (1, 3, H, W) -> mô hình -> softmax
    return pytorch_probabilities, checkpoint["class_names"]  # Trả về hàm tính xác suất và danh sách lớp


if __name__ == '__main__':  # Chỉ chạy khi chạy trực tiếp file
    parser = argparse.ArgumentParser(description="Send images to the OK/NG service and optionally compare with PyTorch")  # Tạo bộ phân tích tham số
    parser.add_argument("paths", nargs="+", help="image files or folders")  # Một hoặc nhiều ảnh/thư mục
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="service address")  # Địa chỉ service
    parser.add_argument("--station-id", default="TEST", help="sent to the service, written in its log")  # Mã trạm gửi kèm
    parser.add_argument("--compare", action="store_true", help="compare every result with PyTorch (needs torch)")  # Bật kiểm chứng
    parser.add_argument("--checkpoint", default=os.path.join(PROJECT_DIR, "outputs", "best.pt"))  # Mô hình PyTorch để so sánh
    args = parser.parse_args()  # Đọc tham số

    session = requests.Session()  # Dùng chung 1 kết nối cho mọi ảnh (nhanh hơn mở kết nối mới mỗi lần)
    try:  # Thử kết nối service
        info = session.get(args.url + "/model-info", timeout=5).json()  # Lấy thông tin mô hình đang chạy trong service
    except requests.exceptions.ConnectionError:  # Service chưa chạy hoặc sai địa chỉ
        print("Cannot connect to {}. Start the service first: python service/app.py".format(args.url))  # Hướng dẫn
        raise SystemExit(1)  # Dừng chương trình
    class_names = info["class_names"]  # Danh sách lớp, vd ['NG', 'OK']
    print("Service model: exported {}, epoch {}, ng_threshold {}".format(info["exported_at"], info["epoch"], info["ng_threshold"]))  # In phiên bản mô hình

    if args.compare:  # Có kiểm chứng
        pytorch_probabilities, pytorch_classes = load_pytorch_reference(args.checkpoint)  # Nạp mô hình PyTorch
        if pytorch_classes != class_names:  # Hai mô hình khác danh sách lớp thì không so được
            print("Class names differ: service {} vs checkpoint {}".format(class_names, pytorch_classes))  # Báo lỗi
            raise SystemExit(1)  # Dừng chương trình
        from evaluate import decide_predictions  # Quy tắc ra kết luận của predict.py

    timings = []  # Thời gian khứ hồi (ms) của từng ảnh
    max_diff = 0.  # Sai lệch xác suất lớn nhất giữa service và PyTorch
    same = compared = 0  # Số ảnh có kết luận trùng / số ảnh đã so
    for image_path in collect_image_paths(args.paths):  # Duyệt từng ảnh
        if not os.path.isfile(image_path):  # File không tồn tại
            print("Skipping {}: file not found".format(image_path))  # Báo rồi bỏ qua
            continue  # Sang ảnh kế tiếp
        start = time.perf_counter()  # Bắt đầu bấm giờ: tính cả gửi ảnh, service xử lý, nhận kết quả
        with open(image_path, "rb") as file:  # Mở ảnh dạng byte
            response = session.post(args.url + "/predict", timeout=10,
                                     files={"file": (os.path.basename(image_path), file)},  # Gửi ảnh (multipart/form-data)
                                     data={"station_id": args.station_id, "product_id": ""})  # Gửi kèm mã trạm
        round_trip_ms = (time.perf_counter() - start) * 1000  # Thời gian khứ hồi (ms)
        if response.status_code != 200:  # Service báo lỗi (vd ảnh hỏng)
            print("Error {} for {}: {}".format(response.status_code, image_path, response.json().get("detail")))  # In lỗi
            continue  # Sang ảnh kế tiếp
        result = response.json()  # Kết quả JSON của service
        timings.append(round_trip_ms)  # Lưu thời gian

        mark = ""  # Ghi chú cuối dòng
        folder = os.path.basename(os.path.dirname(os.path.abspath(image_path)))  # Thư mục cha, vd "NG"
        if folder in class_names and folder != result["result"]:  # Biết nhãn thật và service đoán sai
            mark += "  <- WRONG (true {})".format(folder)  # Đánh dấu giống predict.py
        if args.compare:  # So với PyTorch
            reference = pytorch_probabilities(image_path)  # Xác suất theo PyTorch
            service_probs = [result["probabilities"][name] for name in class_names]  # Xác suất theo service, cùng thứ tự lớp
            max_diff = max(max_diff, max(abs(a - b) for a, b in zip(reference, service_probs)))  # Cập nhật sai lệch lớn nhất
            reference_label = class_names[decide_predictions([reference], class_names, info["ng_threshold"])[0]]  # Kết luận theo PyTorch
            compared += 1  # Đếm số ảnh đã so
            if reference_label == result["result"]:  # Trùng kết luận
                same += 1  # Đếm số ảnh trùng
            else:  # Khác kết luận
                mark += "  <- DIFFERENT FROM PYTORCH ({})".format(reference_label)  # Đánh dấu
        prob_text = "  ".join("{} {:6.2%}".format(name, result["probabilities"][name]) for name in class_names)  # vd "NG  3.10%  OK 96.90%"
        print("{:<4} | {} | service {:5.1f} ms | round trip {:5.1f} ms | {}{}".format(
            result["result"], prob_text, result["inference_ms"], round_trip_ms, image_path, mark))  # In kết quả 1 ảnh

    if timings:  # Có ít nhất 1 ảnh thành công
        print("{} images | round trip average {:.1f} ms | slowest {:.1f} ms".format(
            len(timings), sum(timings) / len(timings), max(timings)))  # Tổng kết tốc độ
    if args.compare and compared:  # Có kiểm chứng
        print("Compare with PyTorch: same decision {}/{} | max probability difference {:.2e} (limit {:.0e})".format(
            same, compared, max_diff, MAX_PROB_DIFF))  # Kết quả kiểm chứng
        if same != compared or max_diff > MAX_PROB_DIFF:  # Không đạt
            print("Compare FAILED. If you retrained, run export_onnx/export.py again, then restart the service.")  # Gợi ý nguyên nhân hay gặp
            raise SystemExit(1)  # Dừng với mã lỗi
        print("Compare passed: the service gives the same results as PyTorch.")  # Đạt

# Service FastAPI phán định OK/NG: WinForms (hoặc chương trình bất kỳ) gửi ảnh lên, service trả kết quả dạng JSON.
# Chỉ dùng ONNX (không cần PyTorch). Mô hình và cách xử lý ảnh lấy từ export_onnx/ (OnnxClassifier), nên kết quả giống hệt predict_onnx.py.
# Chạy: python service/app.py   (mặc định http://127.0.0.1:8000, xem tài liệu API tại http://127.0.0.1:8000/docs)

import argparse  # Thư viện đọc tham số dòng lệnh
import csv  # Thư viện ghi file CSV (log kết quả, mở được bằng Excel)
import datetime  # Thư viện ngày giờ, ghi thời điểm phán định
import io  # Thư viện đọc dữ liệu byte như một file
import os  # Thư viện làm việc với file/thư mục
import sys  # Thư viện hệ thống, dùng để thêm thư mục export_onnx vào đường dẫn import
import threading  # Thư viện khóa (Lock): nhiều yêu cầu cùng lúc không ghi chồng lên file log
import time  # Thư viện đo thời gian xử lý
import uuid  # Thư viện tạo mã ngẫu nhiên không trùng, dùng làm request_id để tra log
from contextlib import asynccontextmanager  # Dùng để viết hàm chạy lúc service khởi động
import uvicorn  # Web server chạy ứng dụng FastAPI
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # Các thành phần của FastAPI
from PIL import Image  # Thư viện mở file ảnh

SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))  # Thư mục service/
EXPORT_DIR = os.path.join(os.path.dirname(SERVICE_DIR), "export_onnx")  # Thư mục export_onnx/ nằm cạnh service/
sys.path.insert(0, EXPORT_DIR)  # Để import predict_onnx.py: chỉ có một nơi chứa cách xử lý ảnh, không chép lại code
from predict_onnx import OnnxClassifier  # Bộ dự đoán ONNX (nạp model.onnx + model.json)

MAX_UPLOAD_MB = 20  # Ảnh lớn hơn mức này bị từ chối (tránh gửi nhầm file khổng lồ làm treo service)
settings = {  # Thiết lập, được ghi đè bởi tham số dòng lệnh trong __main__
    "model_dir": EXPORT_DIR,  # Thư mục chứa model.onnx và model.json
    "log_dir": os.path.join(SERVICE_DIR, "logs"),  # Thư mục lưu log
    "save_images": False,  # True: lưu cả ảnh gốc mỗi lần phán định (tốn ổ cứng, nhớ dọn định kỳ)
}
classifier = None  # Bộ dự đoán, được nạp MỘT lần lúc service khởi động (nạp lại mỗi lần gọi sẽ rất chậm)
log_lock = threading.Lock()  # Khóa để mỗi lúc chỉ một yêu cầu được ghi vào file log


@asynccontextmanager  # Hàm này chạy lúc service khởi động (trước yield) và lúc tắt (sau yield)
async def lifespan(app):  # FastAPI gọi hàm này khi bắt đầu chạy
    global classifier  # Ghi vào biến classifier ở trên
    classifier = OnnxClassifier(settings["model_dir"])  # Nạp mô hình
    classifier.predict_probabilities([Image.new("RGB", (classifier.image_size, classifier.image_size))])  # Chạy thử 1 ảnh đen: yêu cầu đầu tiên không bị chậm
    print("Model loaded: {} (exported {}, epoch {})".format(
        settings["model_dir"], classifier.info["exported_at"], classifier.info["epoch"]))  # Báo đã nạp mô hình
    yield  # Service chạy ở đây cho tới khi bị tắt


app = FastAPI(title="OK/NG inspection service", lifespan=lifespan)  # Tạo ứng dụng FastAPI


def model_version():  # Chuỗi phiên bản mô hình, trả về trong mỗi kết quả để biết ảnh được phán định bằng mô hình nào
    return "{} (epoch {})".format(classifier.info["exported_at"], classifier.info["epoch"])  # vd "2026-10-30 14:36:57 (epoch 10)"


def write_log(result, image_bytes):  # Ghi 1 dòng vào log CSV của ngày hôm nay; nếu bật save_images thì lưu cả ảnh gốc
    os.makedirs(settings["log_dir"], exist_ok=True)  # Tạo thư mục log nếu chưa có
    log_path = os.path.join(settings["log_dir"], "predictions_{}.csv".format(result["timestamp"][:10]))  # Mỗi ngày một file, vd predictions_2026-10-30.csv
    columns = ["timestamp", "request_id", "station_id", "product_id", "filename", "result", "p_ng", "inference_ms", "model_version"]  # Các cột của log
    with log_lock:  # Chỉ một yêu cầu được ghi tại một thời điểm
        is_new_file = not os.path.isfile(log_path)  # File chưa có thì phải ghi dòng tiêu đề
        with open(log_path, "a", newline="", encoding="utf-8-sig") as file:  # Mở để ghi thêm; utf-8-sig để Excel đọc đúng tiếng Việt
            writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")  # Ghi theo tên cột, bỏ qua các trường không có trong columns
            if is_new_file:  # File mới
                writer.writeheader()  # Ghi dòng tiêu đề
            writer.writerow(result)  # Ghi kết quả
    if settings["save_images"]:  # Nếu muốn lưu ảnh để truy vết
        image_dir = os.path.join(settings["log_dir"], "images", result["timestamp"][:10], result["result"])  # vd logs/images/2026-10-30/NG/
        os.makedirs(image_dir, exist_ok=True)  # Tạo thư mục nếu chưa có
        with open(os.path.join(image_dir, "{}_{}".format(result["request_id"], result["filename"])), "wb") as file:  # Tên file có request_id để không trùng
            file.write(image_bytes)  # Lưu nguyên byte ảnh gốc (không nén lại)


@app.get("/health")  # Kiểm tra service còn sống: WinForms gọi lúc khởi động và định kỳ
def health():  # Không cần tham số
    return {"status": "ok", "model_loaded": classifier is not None, "model_version": model_version()}  # Trả về trạng thái


@app.get("/model-info")  # Thông tin mô hình đang chạy (lớp, kích thước ảnh, ngưỡng NG, phiên bản)
def model_info():  # Không cần tham số
    return dict(classifier.info, ng_threshold=classifier.ng_threshold)  # Nội dung model.json


@app.post("/predict")  # Phán định 1 ảnh; viết bằng "def" (không phải "async def") vì dự đoán tốn CPU, FastAPI sẽ chạy nó ở luồng riêng
def predict(file: UploadFile = File(...), station_id: str = Form(""), product_id: str = Form("")):  # Ảnh (bắt buộc), mã trạm và mã sản phẩm (tùy chọn, để truy vết)
    start = time.perf_counter()  # Bắt đầu bấm giờ
    image_bytes = file.file.read()  # Đọc toàn bộ byte của ảnh gửi lên
    if len(image_bytes) > MAX_UPLOAD_MB * 1024 * 1024:  # Ảnh quá lớn
        raise HTTPException(status_code=413, detail="Image larger than {} MB".format(MAX_UPLOAD_MB))  # Trả lỗi 413
    try:  # Thử đọc ảnh
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Giải mã ảnh từ byte, ép về 3 kênh màu
    except OSError:  # Không phải ảnh hoặc ảnh hỏng
        raise HTTPException(status_code=400, detail="Cannot read image: {}".format(file.filename))  # Trả lỗi 400, service vẫn chạy tiếp

    probs = classifier.predict_probabilities([image])[0]  # Xác suất từng lớp
    prediction = classifier.decide(probs)  # Kết luận (chỉ số lớp), cùng quy tắc với predict.py
    result = {  # Kết quả trả về cho WinForms
        "request_id": uuid.uuid4().hex,  # Mã duy nhất của lần phán định này, dùng để tra log
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],  # Thời điểm phán định (đến mili-giây)
        "station_id": station_id,  # Mã trạm gửi ảnh
        "product_id": product_id,  # Mã sản phẩm / mã lô
        "filename": os.path.basename(file.filename or "image"),  # Tên file ảnh (bỏ phần thư mục)
        "result": classifier.class_names[prediction],  # Kết luận: "OK" hoặc "NG"
        "probabilities": {name: round(p, 6) for name, p in zip(classifier.class_names, probs)},  # Xác suất từng lớp, vd {"NG": 0.0312, "OK": 0.9688}
        "p_ng": round(probs[classifier.ng_index], 6) if classifier.ng_index is not None else None,  # Xác suất NG (tiện cho WinForms)
        "ng_threshold": classifier.ng_threshold,  # Ngưỡng NG đang dùng (None = chọn lớp cao nhất)
        "model_version": model_version(),  # Phiên bản mô hình
    }
    result["inference_ms"] = round((time.perf_counter() - start) * 1000, 1)  # Thời gian xử lý trong service: đọc ảnh + xử lý + dự đoán
    write_log(result, image_bytes)  # Ghi log (sau khi đã đo thời gian, để thời gian chỉ tính phần phán định)
    return result  # FastAPI tự đổi dict thành JSON


if __name__ == '__main__':  # Chỉ chạy khi chạy trực tiếp file
    parser = argparse.ArgumentParser(description="OK/NG inspection service (FastAPI + ONNX)")  # Tạo bộ phân tích tham số
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = only this PC; 0.0.0.0 = allow other PCs in the LAN")  # Địa chỉ lắng nghe
    parser.add_argument("--port", type=int, default=8000)  # Cổng
    parser.add_argument("--model-dir", default=settings["model_dir"], help="folder with model.onnx and model.json")  # Thư mục mô hình
    parser.add_argument("--log-dir", default=settings["log_dir"])  # Thư mục log
    parser.add_argument("--save-images", action="store_true", help="also save every received image in LOG_DIR/images")  # Lưu ảnh để truy vết
    args = parser.parse_args()  # Đọc tham số

    if not os.path.isfile(os.path.join(args.model_dir, "model.json")):  # Chưa có mô hình
        print("model.json not found in {}. Run export_onnx/export.py first.".format(args.model_dir))  # Hướng dẫn
        raise SystemExit(1)  # Dừng chương trình
    settings.update(model_dir=args.model_dir, log_dir=args.log_dir, save_images=args.save_images)  # Lưu thiết lập cho lifespan và predict dùng
    uvicorn.run(app, host=args.host, port=args.port)  # Khởi động web server (1 tiến trình, mô hình chỉ nạp 1 lần)

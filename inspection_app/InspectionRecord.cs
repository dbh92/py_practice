using System;  // Kiểu DateTime
using System.Collections.Generic;  // Kiểu Dictionary, List
using System.Text.Json.Serialization;  // Thuộc tính JsonPropertyName: nối tên trường JSON (snake_case) với tên thuộc tính C#

namespace InspectionApp  // Không gian tên của ứng dụng
{
    public class PredictResponse  // Kết quả JSON của POST /predict (xem service/app.py)
    {
        [JsonPropertyName("request_id")] public string RequestId { get; set; }  // Mã duy nhất của lần phán định, dùng để tra log của service
        [JsonPropertyName("timestamp")] public string Timestamp { get; set; }  // Thời điểm service phán định
        [JsonPropertyName("result")] public string Result { get; set; }  // Kết luận: "OK" hoặc "NG"
        [JsonPropertyName("probabilities")] public Dictionary<string, double> Probabilities { get; set; }  // Xác suất từng lớp, vd {"NG": 0.03, "OK": 0.97}
        [JsonPropertyName("p_ng")] public double? PNg { get; set; }  // Xác suất NG (null nếu mô hình không có lớp NG)
        [JsonPropertyName("ng_threshold")] public double? NgThreshold { get; set; }  // Ngưỡng NG đang dùng (null = chọn lớp cao nhất)
        [JsonPropertyName("model_version")] public string ModelVersion { get; set; }  // Phiên bản mô hình
        [JsonPropertyName("inference_ms")] public double InferenceMs { get; set; }  // Thời gian xử lý trong service (ms)
    }

    public class ModelInfo  // Kết quả JSON của GET /model-info (nội dung model.json)
    {
        [JsonPropertyName("model_name")] public string ModelName { get; set; }  // Kiến trúc, vd "resnet18"
        [JsonPropertyName("class_names")] public List<string> ClassNames { get; set; }  // Danh sách lớp, vd ["NG", "OK"]
        [JsonPropertyName("ng_threshold")] public double? NgThreshold { get; set; }  // Ngưỡng NG
        [JsonPropertyName("image_size")] public int ImageSize { get; set; }  // Kích thước ảnh đầu vào của mô hình
        [JsonPropertyName("epoch")] public int Epoch { get; set; }  // Epoch của checkpoint
        [JsonPropertyName("exported_at")] public string ExportedAt { get; set; }  // Thời điểm xuất ONNX
    }

    public class InspectionRecord  // Một lần kiểm tra: hiện ở khung kết quả và thành một dòng trong bảng lịch sử
    {
        public const string Hold = "HOLD";  // Kết quả khi KHÔNG phán định được (service lỗi, timeout...): giữ hàng lại, không cho qua

        public DateTime Time { get; set; }  // Thời điểm bấm kiểm tra (giờ của máy này)
        public string FilePath { get; set; }  // Đường dẫn đầy đủ của ảnh
        public string FileName { get; set; }  // Tên file ảnh
        public string StationId { get; set; }  // Mã trạm
        public string ProductId { get; set; }  // Mã sản phẩm / mã lô
        public string Result { get; set; } = Hold;  // Kết luận; mặc định HOLD để lỗi ở bất kỳ đâu cũng không thành OK
        public double? PNg { get; set; }  // Xác suất NG
        public Dictionary<string, double> Probabilities { get; set; }  // Xác suất từng lớp
        public double? NgThreshold { get; set; }  // Ngưỡng NG service đã dùng
        public double? ServiceMs { get; set; }  // Thời gian xử lý trong service (ms)
        public double RoundTripMs { get; set; }  // Thời gian khứ hồi: đọc ảnh + gửi + service xử lý + nhận kết quả (ms)
        public string RequestId { get; set; }  // Mã tra log của service
        public string ModelVersion { get; set; }  // Phiên bản mô hình
        public string Error { get; set; }  // Lý do HOLD (rỗng nếu phán định thành công)
        public bool? ServiceReached { get; set; }  // true: service đã trả lời; false: không gọi được service; null: chưa gửi (vd không đọc được file ảnh)

        public string TimeText => Time.ToString("HH:mm:ss.fff");  // Giờ hiển thị trong bảng, vd 15:42:24.615
        public string PNgText => PNg.HasValue ? PNg.Value.ToString("P2") : "";  // Xác suất NG dạng phần trăm, vd 73.40%
        public string ServiceMsText => ServiceMs.HasValue ? ServiceMs.Value.ToString("F1") : "";  // Thời gian service, 1 chữ số thập phân
        public string RoundTripMsText => RoundTripMs.ToString("F1");  // Thời gian khứ hồi, 1 chữ số thập phân
    }
}

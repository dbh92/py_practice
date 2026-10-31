using System;  // Kiểu cơ bản (DateTime, TimeSpan, Exception...)
using System.Diagnostics;  // Stopwatch: đo thời gian khứ hồi
using System.IO;  // Đọc file ảnh
using System.Net.Http;  // HttpClient: gửi yêu cầu HTTP tới service
using System.Net.Http.Headers;  // Khai báo kiểu nội dung khi gửi file
using System.Text.Json;  // Đọc JSON trả về
using System.Threading.Tasks;  // Lập trình bất đồng bộ (async/await): giao diện không bị đứng khi chờ service

namespace InspectionApp  // Không gian tên của ứng dụng
{
    public class ServiceClient  // Mọi giao tiếp với service FastAPI nằm ở đây, giao diện chỉ gọi các hàm của lớp này
    {
        private static readonly HttpClient Http = CreateHttpClient();  // CHỈ MỘT HttpClient cho cả chương trình (tạo mới mỗi lần sẽ cạn cổng mạng)

        public string BaseUrl { get; set; } = "http://127.0.0.1:8000";  // Địa chỉ service, đổi được trên giao diện

        private static HttpClient CreateHttpClient()  // Tạo HttpClient dùng chung
        {
            var http = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };  // Chờ tối đa 5 giây mỗi lần gọi
            http.DefaultRequestHeaders.ConnectionClose = true;  // Không giữ kết nối: mỗi lần gửi mở kết nối mới, xong là đóng. Nếu giữ, uvicorn tự đóng kết nối rảnh sau 5 giây, lần gửi sau có thể dùng trúng kết nối vừa bị đóng và ra HOLD giả
            return http;  // Trả về HttpClient đã cấu hình
        }

        public async Task<ModelInfo> GetModelInfoAsync()  // Gọi GET /model-info: thông tin mô hình đang chạy trong service
        {
            string body = await Http.GetStringAsync(BaseUrl + "/model-info");  // Lấy chuỗi JSON (lỗi mạng sẽ ném exception cho nơi gọi xử lý)
            return JsonSerializer.Deserialize<ModelInfo>(body);  // Đổi JSON thành đối tượng ModelInfo
        }

        public async Task<InspectionRecord> InspectAsync(string imagePath, string stationId, string productId)  // Gửi 1 ảnh lên POST /predict; KHÔNG BAO GIỜ ném lỗi: lỗi nào cũng trả về kết quả HOLD
        {
            var record = new InspectionRecord  // Bản ghi kết quả, mặc định Result = HOLD
            {
                Time = DateTime.Now,  // Thời điểm bắt đầu kiểm tra
                FilePath = imagePath,  // Đường dẫn ảnh
                FileName = Path.GetFileName(imagePath),  // Tên file
                StationId = stationId,  // Mã trạm
                ProductId = productId,  // Mã sản phẩm / lô
            };
            var stopwatch = Stopwatch.StartNew();  // Bắt đầu bấm giờ khứ hồi
            try  // Thử gửi ảnh
            {
                byte[] imageBytes = await File.ReadAllBytesAsync(imagePath);  // Đọc nguyên byte ảnh gốc (không nén lại, để không làm mất vết lỗi nhỏ)
                using var form = new MultipartFormDataContent();  // Gói dữ liệu kiểu multipart/form-data, giống form gửi file trên web
                var fileContent = new ByteArrayContent(imageBytes);  // Phần nội dung chứa ảnh
                fileContent.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");  // Khai báo là dữ liệu nhị phân
                form.Add(fileContent, "file", record.FileName);  // Trường "file" (bắt buộc) kèm tên file
                form.Add(new StringContent(stationId ?? ""), "station_id");  // Trường mã trạm
                form.Add(new StringContent(productId ?? ""), "product_id");  // Trường mã sản phẩm
                using var response = await Http.PostAsync(BaseUrl + "/predict", form);  // Gửi và chờ kết quả (tối đa 5 giây)
                record.ServiceReached = true;  // Service đã trả lời (kể cả khi báo lỗi) => vẫn kết nối được
                string body = await response.Content.ReadAsStringAsync();  // Đọc nội dung trả về
                if (!response.IsSuccessStatusCode)  // Service báo lỗi (vd 400 ảnh hỏng, 413 ảnh quá lớn)
                {
                    record.Error = "Service báo lỗi " + (int)response.StatusCode + ": " + ReadErrorDetail(body);  // Ghi lý do, Result vẫn là HOLD
                }
                else  // Thành công
                {
                    var data = JsonSerializer.Deserialize<PredictResponse>(body);  // Đổi JSON thành đối tượng
                    record.Result = data.Result;  // Kết luận OK/NG
                    record.PNg = data.PNg;  // Xác suất NG
                    record.Probabilities = data.Probabilities;  // Xác suất từng lớp
                    record.NgThreshold = data.NgThreshold;  // Ngưỡng NG
                    record.ServiceMs = data.InferenceMs;  // Thời gian xử lý trong service
                    record.RequestId = data.RequestId;  // Mã tra log
                    record.ModelVersion = data.ModelVersion;  // Phiên bản mô hình
                }
            }
            catch (TaskCanceledException)  // Quá 5 giây chưa có kết quả
            {
                record.Error = "Quá thời gian chờ service (timeout)";  // Result vẫn là HOLD
                record.ServiceReached = false;  // Service không trả lời
            }
            catch (HttpRequestException ex)  // Không kết nối được (service chưa chạy, sai địa chỉ, mất mạng)
            {
                record.Error = "Không kết nối được service: " + ex.Message;  // Result vẫn là HOLD
                record.ServiceReached = false;  // Không gọi được service
            }
            catch (IOException ex)  // Không đọc được file ảnh (file đang bị chương trình khác ghi, đã bị xóa...)
            {
                record.Error = "Không đọc được file ảnh: " + ex.Message;  // Result vẫn là HOLD
            }
            catch (Exception ex)  // Mọi lỗi khác (JSON sai định dạng...)
            {
                record.Error = "Lỗi: " + ex.Message;  // Result vẫn là HOLD
            }
            record.RoundTripMs = stopwatch.Elapsed.TotalMilliseconds;  // Dừng bấm giờ
            return record;  // Trả về kết quả (thành công hoặc HOLD)
        }

        private static string ReadErrorDetail(string body)  // Lấy câu báo lỗi trong JSON {"detail": "..."} của FastAPI
        {
            try  // JSON có thể không đúng dạng mong đợi
            {
                using var json = JsonDocument.Parse(body);  // Đọc JSON
                if (json.RootElement.TryGetProperty("detail", out var detail))  // Có trường "detail"
                {
                    return detail.ValueKind == JsonValueKind.String ? detail.GetString() : detail.ToString();  // Chuỗi thì lấy chuỗi, dạng khác (vd lỗi 422) thì lấy nguyên văn
                }
            }
            catch (JsonException)  // Không phải JSON
            {
            }
            return body;  // Trả nguyên văn nội dung
        }
    }
}

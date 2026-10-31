using System;  // Kiểu cơ bản (DateTime, Uri, Exception...)
using System.Collections.Generic;  // Kiểu List
using System.Collections.ObjectModel;  // ObservableCollection: danh sách tự cập nhật lên bảng khi thêm/xóa
using System.ComponentModel;  // Kiểu sự kiện đóng cửa sổ
using System.IO;  // Làm việc với file/thư mục
using System.Linq;  // Lọc, sắp xếp danh sách
using System.Media;  // Âm thanh cảnh báo khi NG/HOLD
using System.Text;  // Mã hóa UTF-8 khi xuất CSV
using System.Threading;  // SemaphoreSlim, CancellationTokenSource
using System.Threading.Tasks;  // Lập trình bất đồng bộ (async/await)
using System.Windows;  // Thư viện giao diện WPF
using System.Windows.Controls;  // Các điều khiển (DataGrid...)
using System.Windows.Media;  // Màu sắc
using System.Windows.Media.Imaging;  // Hiển thị ảnh
using Microsoft.Win32;  // Hộp thoại chọn file, chọn thư mục, lưu file

namespace InspectionApp  // Không gian tên của ứng dụng
{
    public partial class MainWindow : Window  // Cửa sổ chính, phần giao diện khai báo trong MainWindow.xaml
    {
        private static readonly string[] ImageExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp" };  // Các đuôi file được coi là ảnh (giống config.IMAGE_EXTENSIONS)
        private const int MaxHistory = 1000;  // Bảng lịch sử giữ tối đa 1000 dòng, tránh tốn bộ nhớ khi chạy cả ca
        private static readonly Brush OkBrush = new SolidColorBrush(Color.FromRgb(0x2E, 0x7D, 0x32));  // Xanh lá cho OK
        private static readonly Brush NgBrush = new SolidColorBrush(Color.FromRgb(0xC6, 0x28, 0x28));  // Đỏ cho NG
        private static readonly Brush HoldBrush = new SolidColorBrush(Color.FromRgb(0xEF, 0x6C, 0x00));  // Cam cho HOLD
        private static readonly Brush IdleBrush = new SolidColorBrush(Color.FromRgb(0x9C, 0xA3, 0xAF));  // Xám khi chưa có kết quả / đang kiểm tra

        private readonly ServiceClient client = new ServiceClient();  // Đối tượng giao tiếp với service
        private readonly ObservableCollection<InspectionRecord> history = new ObservableCollection<InspectionRecord>();  // Danh sách lịch sử, gắn vào bảng
        private readonly SemaphoreSlim inspectLock = new SemaphoreSlim(1, 1);  // Mỗi lúc chỉ kiểm tra 1 ảnh, các ảnh khác xếp hàng chờ
        private CancellationTokenSource batchCancel;  // Dùng để dừng giữa chừng khi đang kiểm tra nhiều ảnh
        private FileSystemWatcher watcher;  // Theo dõi thư mục: có ảnh mới là tự kiểm tra
        private bool modelInfoLoaded;  // Đã lấy được thông tin mô hình từ service chưa
        private int okCount, ngCount, holdCount;  // Bộ đếm thống kê

        public MainWindow()  // Hàm khởi tạo cửa sổ
        {
            InitializeComponent();  // Tạo các điều khiển khai báo trong MainWindow.xaml
            HistoryGrid.ItemsSource = history;  // Gắn danh sách lịch sử vào bảng
        }

        // ===== Mở / đóng cửa sổ =====

        private async void Window_Loaded(object sender, RoutedEventArgs e)  // Chạy khi cửa sổ vừa hiện lên
        {
            string watchFolder = ReadCommandLine();  // Đọc tham số dòng lệnh (--url, --station, --watch)
            await ConnectAsync();  // Kết nối service lần đầu (sau đó chỉ gọi service khi có ảnh cần kiểm tra)
            if (watchFolder != null)  // Có yêu cầu tự theo dõi thư mục ngay khi mở
            {
                StartWatching(watchFolder);  // Bắt đầu theo dõi
            }
        }

        private void Window_Closing(object sender, CancelEventArgs e)  // Chạy khi đóng cửa sổ
        {
            batchCancel?.Cancel();  // Dừng kiểm tra hàng loạt nếu đang chạy
            watcher?.Dispose();  // Dừng theo dõi thư mục
        }

        private string ReadCommandLine()  // Tham số dòng lệnh, vd: InspectionApp.exe --url http://192.168.1.10:8000 --station ST02 --watch D:\Camera; trả về thư mục cần theo dõi (hoặc null)
        {
            string[] args = Environment.GetCommandLineArgs();  // args[0] là tên chương trình
            string watchFolder = null;  // Thư mục cần theo dõi
            for (int i = 1; i < args.Length - 1; i++)  // Duyệt từng cặp "--tên giá trị"
            {
                if (args[i] == "--url") UrlBox.Text = args[i + 1];  // Địa chỉ service
                else if (args[i] == "--station") StationBox.Text = args[i + 1];  // Mã trạm
                else if (args[i] == "--watch") watchFolder = args[i + 1];  // Thư mục theo dõi
            }
            return watchFolder;  // Trả về thư mục theo dõi
        }

        // ===== Kết nối service =====

        private async void ConnectButton_Click(object sender, RoutedEventArgs e)  // Bấm nút "Kết nối"
        {
            await ConnectAsync();  // Kết nối lại với địa chỉ mới
        }

        private async Task ConnectAsync()  // Lấy thông tin mô hình từ service và cập nhật trạng thái
        {
            client.BaseUrl = UrlBox.Text.Trim().TrimEnd('/');  // Địa chỉ service (bỏ dấu / ở cuối)
            modelInfoLoaded = false;  // Chưa có thông tin mô hình
            ModelText.Text = "Đang kết nối...";  // Báo đang kết nối
            try  // Thử lấy thông tin mô hình
            {
                ModelInfo info = await client.GetModelInfoAsync();  // Gọi GET /model-info
                string threshold = info.NgThreshold.HasValue ? info.NgThreshold.Value.ToString("0.###") : "mặc định";  // Ngưỡng NG để hiển thị
                ModelText.Text = $"Mô hình {info.ModelName} | lớp: {string.Join(", ", info.ClassNames)} | xuất lúc {info.ExportedAt} (epoch {info.Epoch}) | ngưỡng NG: {threshold}";  // Hiện thông tin mô hình
                modelInfoLoaded = true;  // Đã có thông tin
                SetStatus(true);  // Đèn xanh
            }
            catch (Exception ex)  // Không kết nối được
            {
                ModelText.Text = "Không kết nối được service: " + ex.Message;  // Báo lỗi
                SetStatus(false);  // Đèn đỏ
            }
        }

        private void SetStatus(bool connected)  // Cập nhật đèn trạng thái kết nối (theo lần gọi service gần nhất)
        {
            StatusDot.Fill = connected ? Brushes.LimeGreen : Brushes.Red;  // Xanh: kết nối được, đỏ: mất kết nối
            StatusText.Text = connected ? "Đã kết nối" : "Mất kết nối";  // Chữ bên cạnh đèn
        }

        // ===== Chọn ảnh để kiểm tra =====

        private async void OpenFilesButton_Click(object sender, RoutedEventArgs e)  // Bấm "Chọn ảnh..."
        {
            var dialog = new OpenFileDialog  // Hộp thoại chọn file
            {
                Title = "Chọn ảnh cần kiểm tra",  // Tiêu đề
                Filter = "Ảnh|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp|Tất cả file|*.*",  // Chỉ hiện file ảnh
                Multiselect = true,  // Cho chọn nhiều ảnh
            };
            if (dialog.ShowDialog(this) == true)  // Người dùng bấm Open
            {
                await InspectManyAsync(dialog.FileNames.ToList());  // Kiểm tra lần lượt các ảnh đã chọn
            }
        }

        private async void OpenFolderButton_Click(object sender, RoutedEventArgs e)  // Bấm "Kiểm tra cả thư mục..."
        {
            var dialog = new OpenFolderDialog { Title = "Chọn thư mục ảnh cần kiểm tra" };  // Hộp thoại chọn thư mục
            if (dialog.ShowDialog(this) == true)  // Người dùng chọn thư mục
            {
                await InspectManyAsync(ListImages(dialog.FolderName));  // Kiểm tra mọi ảnh trong thư mục
            }
        }

        private void DropArea_DragOver(object sender, DragEventArgs e)  // Đang kéo file qua khung ảnh
        {
            e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop) ? DragDropEffects.Copy : DragDropEffects.None;  // Chỉ nhận file/thư mục
            e.Handled = true;  // Đã xử lý sự kiện
        }

        private async void DropArea_Drop(object sender, DragEventArgs e)  // Thả file/thư mục vào khung ảnh
        {
            if (!(e.Data.GetData(DataFormats.FileDrop) is string[] paths)) return;  // Không phải file thì bỏ qua
            var images = new List<string>();  // Danh sách ảnh cần kiểm tra
            foreach (string path in paths)  // Với từng thứ được thả vào
            {
                if (Directory.Exists(path)) images.AddRange(ListImages(path));  // Thư mục: lấy mọi ảnh bên trong
                else if (IsImage(path)) images.Add(path);  // File ảnh: thêm vào
            }
            await InspectManyAsync(images);  // Kiểm tra lần lượt
        }

        private void StopButton_Click(object sender, RoutedEventArgs e)  // Bấm "Dừng"
        {
            batchCancel?.Cancel();  // Báo dừng; ảnh đang kiểm tra dở sẽ hoàn thành rồi mới dừng
        }

        private static bool IsImage(string path)  // File có phải ảnh không (theo đuôi file)
        {
            return ImageExtensions.Contains(Path.GetExtension(path).ToLowerInvariant());  // So đuôi file với danh sách
        }

        private static List<string> ListImages(string folder)  // Mọi ảnh trong một thư mục, sắp xếp theo tên
        {
            return Directory.GetFiles(folder).Where(IsImage).OrderBy(path => path).ToList();  // Lọc ảnh rồi sắp xếp
        }

        // ===== Theo dõi thư mục (mô phỏng camera lưu ảnh vào một thư mục) =====

        private void WatchButton_Click(object sender, RoutedEventArgs e)  // Bấm "Theo dõi thư mục..." / "Dừng theo dõi"
        {
            if (watcher != null)  // Đang theo dõi
            {
                StopWatching();  // Thì dừng
                return;  // Xong
            }
            var dialog = new OpenFolderDialog { Title = "Chọn thư mục camera lưu ảnh" };  // Hộp thoại chọn thư mục
            if (dialog.ShowDialog(this) == true)  // Người dùng chọn thư mục
            {
                StartWatching(dialog.FolderName);  // Bắt đầu theo dõi
            }
        }

        private void StartWatching(string folder)  // Bắt đầu theo dõi thư mục: có ảnh mới là tự gửi đi kiểm tra
        {
            if (!Directory.Exists(folder))  // Thư mục không tồn tại
            {
                MessageBox.Show("Không tìm thấy thư mục:\n" + folder, "Theo dõi thư mục", MessageBoxButton.OK, MessageBoxImage.Warning);  // Báo lỗi
                return;  // Không theo dõi
            }
            watcher = new FileSystemWatcher(folder) { NotifyFilter = NotifyFilters.FileName };  // Chỉ quan tâm file mới / đổi tên
            watcher.Created += Watcher_FileArrived;  // File mới được tạo
            watcher.Renamed += Watcher_FileArrived;  // File được đổi tên (nhiều phần mềm camera ghi file tạm rồi đổi tên)
            watcher.EnableRaisingEvents = true;  // Bắt đầu theo dõi
            WatchButton.Content = "Dừng theo dõi";  // Đổi chữ trên nút
            WatchText.Text = "Đang theo dõi: " + folder + "  (ảnh mới sẽ tự được kiểm tra)";  // Báo đang theo dõi
            WatchText.Visibility = Visibility.Visible;  // Hiện dòng báo
        }

        private void StopWatching()  // Dừng theo dõi thư mục
        {
            watcher.Dispose();  // Giải phóng bộ theo dõi
            watcher = null;  // Đánh dấu không theo dõi
            WatchButton.Content = "Theo dõi thư mục...";  // Trả lại chữ trên nút
            WatchText.Visibility = Visibility.Collapsed;  // Ẩn dòng báo
        }

        private void Watcher_FileArrived(object sender, FileSystemEventArgs e)  // Có file mới trong thư mục theo dõi (chạy ở luồng nền)
        {
            if (!IsImage(e.FullPath)) return;  // Không phải ảnh thì bỏ qua
            Dispatcher.InvokeAsync(async () =>  // Chuyển về luồng giao diện (chỉ luồng giao diện được sửa điều khiển)
            {
                await WaitUntilFileReadyAsync(e.FullPath);  // Chờ camera ghi xong file
                await InspectOneAsync(e.FullPath);  // Kiểm tra ảnh
            });
        }

        private static async Task WaitUntilFileReadyAsync(string path)  // Chờ tối đa 2 giây cho tới khi file không còn bị chương trình khác ghi
        {
            for (int attempt = 0; attempt < 20; attempt++)  // Thử tối đa 20 lần
            {
                try  // Thử mở file độc quyền
                {
                    using (File.Open(path, FileMode.Open, FileAccess.Read, FileShare.None)) return;  // Mở được = đã ghi xong
                }
                catch (IOException)  // File đang bị ghi
                {
                    await Task.Delay(100);  // Chờ 0,1 giây rồi thử lại
                }
            }
        }  // Hết 2 giây vẫn chưa mở được thì cứ kiểm tra: lỗi đọc file sẽ thành HOLD

        // ===== Kiểm tra ảnh =====

        private async Task InspectManyAsync(List<string> paths)  // Kiểm tra lần lượt nhiều ảnh, có thể bấm "Dừng" giữa chừng
        {
            if (paths.Count == 0)  // Không có ảnh nào
            {
                ProgressText.Text = "Không có ảnh nào để kiểm tra.";  // Báo
                return;  // Xong
            }
            batchCancel = new CancellationTokenSource();  // Tạo cờ dừng mới
            SetBatchButtons(running: true);  // Khóa các nút chọn ảnh, mở nút Dừng
            try  // Đảm bảo luôn mở lại các nút dù có lỗi
            {
                for (int i = 0; i < paths.Count; i++)  // Với từng ảnh
                {
                    if (batchCancel.IsCancellationRequested)  // Người dùng bấm Dừng
                    {
                        ProgressText.Text = $"Đã dừng sau {i}/{paths.Count} ảnh.";  // Báo đã dừng
                        return;  // Thoát
                    }
                    ProgressText.Text = $"Đang kiểm tra {i + 1}/{paths.Count}...";  // Báo tiến độ
                    await InspectOneAsync(paths[i]);  // Kiểm tra ảnh
                }
                ProgressText.Text = $"Xong {paths.Count} ảnh.";  // Báo hoàn thành
            }
            finally  // Luôn chạy
            {
                SetBatchButtons(running: false);  // Mở lại các nút
                batchCancel = null;  // Xóa cờ dừng
            }
        }

        private void SetBatchButtons(bool running)  // Bật/tắt các nút khi đang kiểm tra hàng loạt
        {
            OpenFilesButton.IsEnabled = !running;  // Đang chạy thì không cho chọn thêm
            OpenFolderButton.IsEnabled = !running;  // Đang chạy thì không cho chọn thêm
            StopButton.IsEnabled = running;  // Chỉ bấm Dừng được khi đang chạy
        }

        private async Task InspectOneAsync(string path)  // Kiểm tra 1 ảnh: hiện ảnh, gửi lên service, hiện kết quả, thêm vào lịch sử
        {
            await inspectLock.WaitAsync();  // Xếp hàng: chờ ảnh trước kiểm tra xong
            try  // Đảm bảo luôn nhả khóa
            {
                ShowImage(path);  // Hiện ảnh
                ShowBusy();  // Khung kết quả chuyển sang "Đang kiểm tra"
                InspectionRecord record = await client.InspectAsync(path, StationBox.Text.Trim(), ProductBox.Text.Trim());  // Gửi lên service (lỗi nào cũng thành HOLD)
                if (record.ServiceReached.HasValue) SetStatus(record.ServiceReached.Value);  // Đèn theo kết quả lần gửi này (không gửi được vì lỗi file thì giữ nguyên đèn)
                if (record.ServiceReached == true && !modelInfoLoaded) await ConnectAsync();  // Service vừa chạy lại mà chưa có thông tin mô hình thì lấy lại
                AddToHistory(record);  // Thêm vào bảng và cập nhật thống kê
                ShowRecord(record);  // Hiện kết quả
                if (record.Result != "OK") SystemSounds.Exclamation.Play();  // NG hoặc HOLD: kêu cảnh báo
            }
            finally  // Luôn chạy
            {
                inspectLock.Release();  // Nhả khóa cho ảnh tiếp theo
            }
        }

        // ===== Hiển thị =====

        private void ShowImage(string path)  // Hiện ảnh trong khung bên trái
        {
            ImageNameText.Text = Path.GetFileName(path);  // Tên file ở góc trên
            ImageNameBox.Visibility = Visibility.Visible;  // Hiện ô tên file
            try  // Ảnh có thể hỏng hoặc đã bị xóa
            {
                var bitmap = new BitmapImage();  // Đối tượng ảnh của WPF
                bitmap.BeginInit();  // Bắt đầu thiết lập
                bitmap.CacheOption = BitmapCacheOption.OnLoad;  // Đọc hết vào bộ nhớ rồi đóng file ngay (không giữ khóa file của camera)
                bitmap.CreateOptions = BitmapCreateOptions.IgnoreImageCache;  // Luôn đọc lại từ ổ đĩa (file cùng tên có thể đã bị ghi đè)
                bitmap.UriSource = new Uri(Path.GetFullPath(path));  // Đường dẫn ảnh
                bitmap.EndInit();  // Đọc ảnh
                bitmap.Freeze();  // Khóa ảnh lại: nhẹ hơn và an toàn khi dùng chung
                PreviewImage.Source = bitmap;  // Hiện ảnh
                PlaceholderText.Visibility = Visibility.Collapsed;  // Ẩn dòng hướng dẫn kéo thả
            }
            catch (Exception)  // Không đọc được ảnh
            {
                PreviewImage.Source = null;  // Xóa ảnh cũ
                PlaceholderText.Text = "Không hiển thị được ảnh này";  // Báo
                PlaceholderText.Visibility = Visibility.Visible;  // Hiện dòng báo
            }
        }

        private void ShowBusy()  // Khung kết quả trong lúc chờ service
        {
            ResultBanner.Background = IdleBrush;  // Nền xám
            ResultText.Text = "...";  // Chữ lớn
            ResultSubText.Text = "Đang kiểm tra";  // Chữ nhỏ
        }

        private void ShowRecord(InspectionRecord record)  // Hiện chi tiết một kết quả ở bên phải
        {
            ResultText.Text = record.Result;  // OK / NG / HOLD
            if (record.Result == "OK")  // Đạt
            {
                ResultBanner.Background = OkBrush;  // Nền xanh
                ResultSubText.Text = "Đạt";  // Chữ nhỏ
            }
            else if (record.Result == InspectionRecord.Hold)  // Không phán định được
            {
                ResultBanner.Background = HoldBrush;  // Nền cam
                ResultSubText.Text = "Không phán định được - giữ lại kiểm tra";  // Nhắc không cho hàng qua
            }
            else  // NG (hoặc tên một loại lỗi nếu mô hình có nhiều lớp lỗi)
            {
                ResultBanner.Background = NgBrush;  // Nền đỏ
                ResultSubText.Text = "Lỗi - loại sản phẩm";  // Chữ nhỏ
            }
            PNgBar.Value = record.PNg ?? 0;  // Thanh xác suất NG
            PNgValueText.Text = record.PNgText;  // Số phần trăm trên thanh
            ProbabilitiesText.Text = record.Probabilities == null ? "" :
                string.Join("   ", record.Probabilities.Select(p => $"{p.Key} {p.Value:P2}"));  // vd "NG 73.40%   OK 26.60%"
            ServiceMsText.Text = record.ServiceMs.HasValue ? record.ServiceMsText + " ms" : "";  // Thời gian trong service
            RoundTripText.Text = record.RoundTripMsText + " ms";  // Thời gian khứ hồi
            ThresholdText.Text = record.Result == InspectionRecord.Hold ? "" :
                (record.NgThreshold.HasValue ? record.NgThreshold.Value.ToString("0.###") : "mặc định (chọn lớp cao nhất)");  // Ngưỡng NG
            VersionText.Text = record.ModelVersion ?? "";  // Phiên bản mô hình
            RequestIdText.Text = record.RequestId ?? "";  // Mã tra log (chọn và copy được)
            ErrorText.Text = record.Error ?? "";  // Lý do HOLD
        }

        private void AddToHistory(InspectionRecord record)  // Thêm kết quả vào đầu bảng lịch sử và cập nhật thống kê
        {
            history.Insert(0, record);  // Mới nhất ở trên cùng
            if (history.Count > MaxHistory) history.RemoveAt(history.Count - 1);  // Quá giới hạn thì bỏ dòng cũ nhất
            if (record.Result == "OK") okCount++;  // Đếm OK
            else if (record.Result == InspectionRecord.Hold) holdCount++;  // Đếm HOLD
            else ngCount++;  // Đếm NG (mọi lớp lỗi)
            UpdateCounters();  // Cập nhật số trên màn hình
        }

        private void UpdateCounters()  // Cập nhật khung thống kê
        {
            int total = okCount + ngCount + holdCount;  // Tổng số lần kiểm tra
            TotalCountText.Text = total.ToString();  // Tổng
            OkCountText.Text = okCount.ToString();  // Số OK
            NgCountText.Text = ngCount.ToString();  // Số NG
            HoldCountText.Text = holdCount.ToString();  // Số HOLD
            NgRateText.Text = total == 0 ? "0%" : ((double)ngCount / total).ToString("P1");  // Tỉ lệ NG trên tổng số
        }

        private void HistoryGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)  // Bấm vào một dòng lịch sử
        {
            if (!(HistoryGrid.SelectedItem is InspectionRecord record)) return;  // Không chọn dòng nào thì thôi
            ShowImage(record.FilePath);  // Hiện lại ảnh
            ShowRecord(record);  // Hiện lại kết quả
        }

        // ===== Lịch sử: xuất CSV, xóa =====

        private void ExportButton_Click(object sender, RoutedEventArgs e)  // Bấm "Xuất CSV..."
        {
            if (history.Count == 0)  // Chưa có gì để xuất
            {
                MessageBox.Show("Chưa có kết quả nào.", "Xuất CSV", MessageBoxButton.OK, MessageBoxImage.Information);  // Báo
                return;  // Xong
            }
            var dialog = new SaveFileDialog  // Hộp thoại lưu file
            {
                Title = "Lưu lịch sử kiểm tra",  // Tiêu đề
                Filter = "CSV (mở bằng Excel)|*.csv",  // Chỉ lưu CSV
                FileName = "inspection_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".csv",  // Tên mặc định có ngày giờ
            };
            if (dialog.ShowDialog(this) != true) return;  // Người dùng bấm Cancel
            var lines = new List<string> { "time,station_id,product_id,file,result,p_ng,service_ms,round_trip_ms,request_id,model_version,error" };  // Dòng tiêu đề
            foreach (InspectionRecord r in history.Reverse())  // Từ cũ đến mới
            {
                lines.Add(string.Join(",", new[]  // Nối các cột bằng dấu phẩy
                {
                    r.Time.ToString("yyyy-MM-dd HH:mm:ss.fff"), r.StationId, r.ProductId, r.FilePath, r.Result,
                    r.PNg.HasValue ? r.PNg.Value.ToString("0.######") : "", r.ServiceMsText, r.RoundTripMsText,
                    r.RequestId, r.ModelVersion, r.Error,
                }.Select(CsvField)));  // Bọc từng ô cho đúng chuẩn CSV
            }
            File.WriteAllLines(dialog.FileName, lines, new UTF8Encoding(true));  // UTF-8 có BOM để Excel đọc đúng tiếng Việt
            ProgressText.Text = "Đã lưu " + dialog.FileName;  // Báo đã lưu
        }

        private static string CsvField(string value)  // Bọc một ô CSV: ô có dấu phẩy, ngoặc kép, xuống dòng thì phải đặt trong ngoặc kép
        {
            value ??= "";  // null thành chuỗi rỗng
            if (value.IndexOfAny(new[] { ',', '"', '\n', '\r' }) < 0) return value;  // Không có ký tự đặc biệt thì giữ nguyên
            return "\"" + value.Replace("\"", "\"\"") + "\"";  // Nhân đôi ngoặc kép bên trong rồi bọc ngoặc kép
        }

        private void ClearButton_Click(object sender, RoutedEventArgs e)  // Bấm "Xóa lịch sử"
        {
            if (MessageBox.Show("Xóa toàn bộ lịch sử và thống kê trên màn hình?\n(Log của service không bị ảnh hưởng)", "Xóa lịch sử",
                    MessageBoxButton.YesNo, MessageBoxImage.Question) != MessageBoxResult.Yes) return;  // Hỏi lại cho chắc
            history.Clear();  // Xóa bảng
            okCount = ngCount = holdCount = 0;  // Đặt lại bộ đếm
            UpdateCounters();  // Cập nhật màn hình
        }
    }
}

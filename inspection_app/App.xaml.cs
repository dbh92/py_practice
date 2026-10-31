using System.Windows;  // Thư viện giao diện WPF
using System.Windows.Threading;  // Kiểu sự kiện lỗi của luồng giao diện

namespace InspectionApp  // Không gian tên của ứng dụng
{
    public partial class App : Application  // Lớp ứng dụng, phần còn lại khai báo trong App.xaml
    {
        private void App_DispatcherUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs e)  // Bắt mọi lỗi chưa được xử lý trên giao diện
        {
            MessageBox.Show("Lỗi không mong muốn:\n" + e.Exception.Message, "Lỗi", MessageBoxButton.OK, MessageBoxImage.Error);  // Báo lỗi cho người dùng
            e.Handled = true;  // Đánh dấu đã xử lý: ứng dụng không bị tắt đột ngột giữa ca làm việc
        }
    }
}

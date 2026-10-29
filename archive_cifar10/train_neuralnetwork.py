import argparse  # Thư viện đọc tham số dòng lệnh (--epochs, --lr, ...)
import os  # Thư viện làm việc với file/thư mục
import torch  # Thư viện PyTorch chính
import torch.nn as nn  # Module chứa các lớp mạng và hàm mất mát
from torch.utils.data import DataLoader, Subset  # DataLoader: chia batch; Subset: lấy một phần bộ dữ liệu theo danh sách chỉ số
from sklearn.metrics import classification_report  # Hàm in báo cáo precision/recall/F1 cho từng lớp
from dataset import CIFARDataset, get_transforms, load_class_names  # Các thành phần dữ liệu tự viết trong dataset.py
from models import MODELS  # Bảng tra các mô hình trong models.py


def get_args():  # Định nghĩa và đọc các tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 classifier")  # Tạo bộ phân tích tham số
    parser.add_argument("--model", choices=MODELS.keys(), default="cnn")  # Chọn mô hình: "mlp" hoặc "cnn" (mặc định cnn)
    parser.add_argument("--epochs", type=int, default=100)  # Số epoch (số lần duyệt hết tập train)
    parser.add_argument("--batch-size", type=int, default=64)  # Số ảnh trong một batch
    parser.add_argument("--lr", type=float, default=1e-2)  # Learning rate ban đầu (tốc độ học)
    parser.add_argument("--momentum", type=float, default=0.9)  # Momentum của SGD: giữ "quán tính" giúp hội tụ nhanh và mượt hơn
    parser.add_argument("--weight-decay", type=float, default=5e-4)  # Weight decay: phạt trọng số lớn để chống overfit
    parser.add_argument("--val-size", type=int, default=5000)  # Số ảnh tách từ tập train ra làm validation
    # Dữ liệu đã nằm sẵn trong RAM; trên Windows mỗi worker phải spawn và copy dataset nên 0 thường nhanh hơn
    parser.add_argument("--num-workers", type=int, default=0)  # Số tiến trình phụ để nạp dữ liệu (0 = nạp ngay trong tiến trình chính)
    parser.add_argument("--seed", type=int, default=42)  # Hạt giống ngẫu nhiên để kết quả lặp lại được giữa các lần chạy
    parser.add_argument("--data-root", default="./data")  # Thư mục chứa dữ liệu
    parser.add_argument("--checkpoint-dir", default="./checkpoints")  # Thư mục lưu checkpoint (trọng số mô hình)
    parser.add_argument("--resume", action="store_true", help="continue from <checkpoint-dir>/<model>_last.pt")  # Có cờ này thì chạy tiếp từ checkpoint cuối
    return parser.parse_args()  # Đọc tham số người dùng nhập và trả về đối tượng args (vd: args.epochs)


def make_loader(dataset, args, shuffle):  # Hàm tiện ích tạo DataLoader với cấu hình chung
    return DataLoader(  # Tạo DataLoader
        dataset=dataset,  # Bộ dữ liệu nguồn
        batch_size=args.batch_size,  # Kích thước batch
        shuffle=shuffle,  # Có xáo trộn dữ liệu mỗi epoch không (True cho train, False cho val/test)
        num_workers=args.num_workers,  # Số tiến trình phụ nạp dữ liệu
        persistent_workers=args.num_workers > 0,  # Giữ worker sống qua các epoch để khỏi phải tạo lại (chỉ hợp lệ khi có worker)
        pin_memory=torch.cuda.is_available(),  # Khi có GPU: đặt dữ liệu vào vùng nhớ cố định để chép sang GPU nhanh hơn
        drop_last=shuffle,  # Train: bỏ batch cuối nếu bị thiếu ảnh; val/test: giữ lại để đánh giá đủ mọi ảnh
    )


def evaluate(model, dataloader, device):  # Đánh giá mô hình trên một bộ dữ liệu, trả về (accuracy, dự đoán, nhãn thật)
    model.eval()  # Chuyển sang chế độ đánh giá: tắt Dropout, BatchNorm dùng thống kê đã học
    all_predictions = []  # Danh sách dự đoán của mọi ảnh
    all_labels = []  # Danh sách nhãn thật của mọi ảnh
    with torch.no_grad():  # Không tính gradient khi đánh giá -> nhanh hơn và tốn ít bộ nhớ hơn
        for images, labels in dataloader:  # Duyệt từng batch
            outputs = model(images.to(device))  # Chuyển ảnh sang thiết bị và cho qua mô hình -> logits (batch, 10)
            # Chọn lớp có điểm dự đoán cao nhất cho mỗi ảnh
            all_predictions.extend(outputs.argmax(dim=1).cpu().tolist())  # argmax theo chiều lớp -> chỉ số lớp; đưa về CPU rồi đổi sang list
            all_labels.extend(labels.tolist())  # Thêm nhãn thật của batch (labels vẫn ở CPU)
    accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)  # Độ chính xác = số dự đoán đúng / tổng số ảnh
    return accuracy, all_predictions, all_labels  # Trả về kết quả


if __name__ == '__main__':  # Bắt buộc trên Windows khi dùng num_workers > 0; chỉ chạy khi chạy trực tiếp file
    args = get_args()  # Đọc tham số dòng lệnh
    torch.manual_seed(args.seed)  # Cố định hạt giống ngẫu nhiên (khởi tạo trọng số, xáo trộn, augmentation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Dùng GPU nếu có, không thì CPU

    # Tách validation từ tập train (không augment) để tập test chỉ dùng một lần ở cuối
    full_train = CIFARDataset(root=args.data_root, train=True, transform=get_transforms(train=True))  # 50.000 ảnh train, CÓ augmentation
    full_val = CIFARDataset(root=args.data_root, train=True, transform=get_transforms(train=False))  # Cùng 50.000 ảnh đó nhưng KHÔNG augmentation
    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(args.seed)).tolist()  # Hoán vị ngẫu nhiên các chỉ số 0..49999 (cố định theo seed)
    train_dataset = Subset(full_train, indices[args.val_size:])  # 45.000 chỉ số sau dùng để train
    val_dataset = Subset(full_val, indices[:args.val_size])  # 5.000 chỉ số đầu dùng làm validation (không trùng với train)
    test_dataset = CIFARDataset(root=args.data_root, train=False, transform=get_transforms(train=False))  # 10.000 ảnh test, chỉ chuẩn hóa

    train_dataloader = make_loader(train_dataset, args, shuffle=True)  # DataLoader train: có xáo trộn
    val_dataloader = make_loader(val_dataset, args, shuffle=False)  # DataLoader validation: không xáo trộn
    test_dataloader = make_loader(test_dataset, args, shuffle=False)  # DataLoader test: không xáo trộn

    model = MODELS[args.model](num_classes=10).to(device)  # Tra tên mô hình trong bảng, tạo mô hình và chuyển sang thiết bị
    criterion = nn.CrossEntropyLoss()  # Hàm mất mát cho bài toán phân loại nhiều lớp (đã gồm softmax bên trong)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)  # Bộ tối ưu SGD: cập nhật trọng số theo gradient
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # Giảm learning rate dần theo hình cosine, về gần 0 ở epoch cuối

    os.makedirs(args.checkpoint_dir, exist_ok=True)  # Tạo thư mục checkpoint nếu chưa có (exist_ok=True: có rồi thì không báo lỗi)
    last_path = os.path.join(args.checkpoint_dir, "{}_last.pt".format(args.model))  # Đường dẫn checkpoint của epoch gần nhất (vd: cnn_last.pt)
    best_path = os.path.join(args.checkpoint_dir, "{}_best.pt".format(args.model))  # Đường dẫn checkpoint tốt nhất theo validation (vd: cnn_best.pt)
    start_epoch = 0  # Epoch bắt đầu (0 nếu train từ đầu)
    best_accuracy = 0.  # Độ chính xác validation tốt nhất đến hiện tại
    if args.resume and os.path.isfile(last_path):  # Nếu có cờ --resume và file checkpoint tồn tại
        checkpoint = torch.load(last_path, map_location=device)  # Nạp checkpoint; map_location đưa tensor về đúng thiết bị hiện tại
        model.load_state_dict(checkpoint["model"])  # Khôi phục trọng số mô hình
        optimizer.load_state_dict(checkpoint["optimizer"])  # Khôi phục trạng thái optimizer (gồm momentum)
        scheduler.load_state_dict(checkpoint["scheduler"])  # Khôi phục lịch learning rate
        start_epoch = checkpoint["epoch"]  # Tiếp tục từ epoch đã lưu
        best_accuracy = checkpoint["best_accuracy"]  # Khôi phục độ chính xác tốt nhất
        print("Resumed from epoch {}, best val accuracy {:.4f}".format(start_epoch, best_accuracy))  # Thông báo đã chạy tiếp

    print("Model: {} | device: {} | train/val/test: {}/{}/{}".format(
        args.model, device, len(train_dataset), len(val_dataset), len(test_dataset)))  # In thông tin cấu hình trước khi train
    for epoch in range(start_epoch, args.epochs):  # Vòng lặp chính: mỗi vòng là một epoch
        model.train()  # Chuyển sang chế độ train: bật Dropout, BatchNorm cập nhật thống kê
        total_loss = 0.  # Cộng dồn loss của các batch để tính loss trung bình của epoch
        for images, labels in train_dataloader:  # Duyệt từng batch của tập train
            images = images.to(device)  # Chuyển ảnh sang thiết bị
            labels = labels.to(device)  # Chuyển nhãn sang thiết bị
            # forward
            outputs = model(images)  # Lan truyền xuôi: tính logits (batch, 10)
            loss_value = criterion(outputs, labels)  # Tính loss giữa dự đoán và nhãn thật
            # backward
            optimizer.zero_grad()  # Xóa gradient cũ (PyTorch mặc định cộng dồn gradient)
            loss_value.backward()  # Lan truyền ngược: tính gradient của loss theo từng trọng số
            optimizer.step()  # Cập nhật trọng số theo gradient vừa tính
            total_loss += loss_value.item()  # .item() lấy giá trị số Python từ tensor loss rồi cộng dồn
        scheduler.step()  # Hết một epoch: cập nhật learning rate theo lịch cosine

        val_accuracy, _, _ = evaluate(model, val_dataloader, device)  # Đánh giá trên validation (bỏ qua danh sách dự đoán/nhãn)
        is_best = val_accuracy > best_accuracy  # Kiểm tra epoch này có tốt nhất từ trước tới giờ không
        best_accuracy = max(best_accuracy, val_accuracy)  # Cập nhật độ chính xác tốt nhất
        print("Epoch {}/{} | loss {:.4f} | val accuracy {:.4f}{}".format(
            epoch + 1, args.epochs, total_loss / len(train_dataloader), val_accuracy, " *" if is_best else ""))  # In kết quả epoch; dấu * = tốt nhất mới

        if is_best:  # Nếu là kết quả tốt nhất
            torch.save(model.state_dict(), best_path)  # Lưu trọng số mô hình vào file best
        torch.save({  # Luôn lưu checkpoint đầy đủ của epoch hiện tại để có thể --resume
            "epoch": epoch + 1,  # Số epoch đã hoàn thành
            "model": model.state_dict(),  # Trọng số mô hình
            "optimizer": optimizer.state_dict(),  # Trạng thái optimizer
            "scheduler": scheduler.state_dict(),  # Trạng thái lịch learning rate
            "best_accuracy": best_accuracy,  # Độ chính xác validation tốt nhất
        }, last_path)  # Ghi vào file last

    # Đánh giá trên tập test đúng một lần, với mô hình tốt nhất theo validation
    model.load_state_dict(torch.load(best_path, map_location=device))  # Nạp lại trọng số của epoch tốt nhất
    test_accuracy, all_predictions, all_labels = evaluate(model, test_dataloader, device)  # Đánh giá trên tập test
    print("Test accuracy (best val checkpoint): {:.4f}".format(test_accuracy))  # In độ chính xác trên test
    print(classification_report(all_labels, all_predictions, target_names=load_class_names(args.data_root),
                                digits=4))  # In precision/recall/F1 từng lớp, kèm tên lớp, làm tròn 4 chữ số

import argparse  # Thư viện đọc tham số dòng lệnh
import os  # Thư viện làm việc với file/thư mục
import random  # Thư viện tạo số ngẫu nhiên, dùng để xáo trộn ảnh trước khi chia
import shutil  # Thư viện sao chép/xóa file và thư mục
import config  # Các thiết lập của dự án (config.py)


def list_images(folder):  # Trả về danh sách tên file ảnh trong một thư mục, sắp xếp theo tên
    return sorted(name for name in os.listdir(folder) if name.lower().endswith(config.IMAGE_EXTENSIONS))  # Chỉ lấy file có đuôi ảnh


if __name__ == '__main__':  # Chỉ chạy khi chạy trực tiếp file
    parser = argparse.ArgumentParser(description="Split data/raw/<class>/ into data/split/train|val|test/<class>/")  # Tạo bộ phân tích tham số
    parser.add_argument("--overwrite", action="store_true", help="delete the existing split folder and split again")  # Cho phép chia lại khi đã chia rồi
    args = parser.parse_args()  # Đọc tham số

    if not os.path.isdir(config.RAW_DIR):  # Chưa có thư mục ảnh gốc
        print("Folder not found: {}. Put your images in {}/OK and {}/NG first.".format(
            config.RAW_DIR, config.RAW_DIR, config.RAW_DIR))  # Hướng dẫn người dùng
        raise SystemExit(1)  # Dừng chương trình
    class_names = sorted(name for name in os.listdir(config.RAW_DIR)
                         if os.path.isdir(os.path.join(config.RAW_DIR, name)))  # Mỗi thư mục con trong data/raw là một lớp (vd: NG, OK)
    if len(class_names) < 2:  # Phân loại cần ít nhất 2 lớp
        print("Need at least 2 class folders in {}, found: {}".format(config.RAW_DIR, class_names))  # Báo lỗi
        raise SystemExit(1)  # Dừng chương trình

    images_per_class = {name: list_images(os.path.join(config.RAW_DIR, name)) for name in class_names}  # Danh sách ảnh của từng lớp
    empty_classes = [name for name, images in images_per_class.items() if not images]  # Các lớp chưa có ảnh nào
    if empty_classes:  # Kiểm tra trước khi tạo/xóa bất cứ thứ gì
        print("No images in: {}. Put images into {}/<class>/ first.".format(empty_classes, config.RAW_DIR))  # Hướng dẫn
        raise SystemExit(1)  # Dừng chương trình

    if os.path.exists(config.SPLIT_DIR):  # Nếu đã chia trước đó
        if not args.overwrite:  # Không cho phép ghi đè thì dừng, tránh vô tình làm thay đổi tập test
            print("{} already exists. Run again with --overwrite to delete it and split again.".format(config.SPLIT_DIR))  # Hướng dẫn
            raise SystemExit(1)  # Dừng chương trình
        shutil.rmtree(config.SPLIT_DIR)  # Xóa thư mục chia cũ (ảnh gốc trong data/raw không bị ảnh hưởng)

    rng = random.Random(config.SEED)  # Bộ sinh ngẫu nhiên riêng với seed cố định -> lần nào chạy cũng chia giống nhau
    print("Classes: {}".format(class_names))  # In danh sách lớp
    for class_name in class_names:  # Chia riêng từng lớp để tỉ lệ OK/NG trong train, val, test giống nhau
        images = images_per_class[class_name]  # Danh sách ảnh của lớp này
        rng.shuffle(images)  # Xáo trộn ngẫu nhiên thứ tự ảnh
        num_test = round(len(images) * config.TEST_RATIO)  # Số ảnh cho tập test
        num_val = round(len(images) * config.VAL_RATIO)  # Số ảnh cho tập validation
        splits = {  # Cắt danh sách đã xáo trộn thành 3 phần không trùng nhau
            "test": images[:num_test],  # Phần đầu: test
            "val": images[num_test:num_test + num_val],  # Phần giữa: validation
            "train": images[num_test + num_val:],  # Phần còn lại: train
        }
        for split, names in splits.items():  # Với từng tập (train, val, test)
            target_dir = os.path.join(config.SPLIT_DIR, split, class_name)  # Thư mục đích, vd: data/split/train/NG
            os.makedirs(target_dir, exist_ok=True)  # Tạo thư mục nếu chưa có
            for name in names:  # Với từng ảnh thuộc tập này
                shutil.copy2(os.path.join(config.RAW_DIR, class_name, name), os.path.join(target_dir, name))  # Sao chép (không di chuyển) ảnh gốc sang
        print("{:<12} total {:>5} | train {:>5} | val {:>4} | test {:>4}".format(
            class_name, len(images), len(splits["train"]), len(splits["val"]), len(splits["test"])))  # In số ảnh mỗi tập
        if min(len(names) for names in splits.values()) == 0:  # Có tập nào không có ảnh nào của lớp này
            print("  Warning: class '{}' has too few images, some splits are empty. Collect more images.".format(class_name))  # Cảnh báo
    print("Done. Images copied to {}".format(config.SPLIT_DIR))  # Báo hoàn thành

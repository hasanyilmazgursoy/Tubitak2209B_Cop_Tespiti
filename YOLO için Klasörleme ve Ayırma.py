import os
import glob
import shutil
import random

# 📌 Mevcut veri seti (final_dataset içindeki görseller ve etiketler)
final_images_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\final_dataset\images"
final_labels_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\final_dataset\labels"

# 📌 YOLO için yeni klasörler
yolo_base_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\final_yolo_dataset"
images_train = os.path.join(yolo_base_dir, "images/train")
images_val = os.path.join(yolo_base_dir, "images/val")
images_test = os.path.join(yolo_base_dir, "images/test")
labels_train = os.path.join(yolo_base_dir, "labels/train")
labels_val = os.path.join(yolo_base_dir, "labels/val")
labels_test = os.path.join(yolo_base_dir, "labels/test")

# 📌 Klasörleri oluştur
for folder in [images_train, images_val, images_test, labels_train, labels_val, labels_test]:
    os.makedirs(folder, exist_ok=True)

# 📌 Tüm resimleri listele
image_files = glob.glob(os.path.join(final_images_dir, "*.jpg"))

# 📌 Veri setini %80 train, %10 val, %10 test olarak bölelim
random.shuffle(image_files)
num_images = len(image_files)
train_split = int(0.8 * num_images)
val_split = int(0.9 * num_images)  # %80'den sonrası val, %90'dan sonrası test olacak

train_files = image_files[:train_split]
val_files = image_files[train_split:val_split]
test_files = image_files[val_split:]

# 📌 Dosyaları ilgili klasörlere taşı
def move_files(file_list, target_img_folder, target_lbl_folder):
    for img_path in file_list:
        img_name = os.path.basename(img_path)
        lbl_path = os.path.join(final_labels_dir, img_name.replace(".jpg", ".txt"))

        # 📌 Resmi taşı
        shutil.copy(img_path, os.path.join(target_img_folder, img_name))

        # 📌 Etiket varsa taşı
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(target_lbl_folder, img_name.replace(".jpg", ".txt")))

# 📌 Train, Val, Test klasörlerine kopyala
move_files(train_files, images_train, labels_train)
move_files(val_files, images_val, labels_val)
move_files(test_files, images_test, labels_test)

print("✅ YOLO veri seti başarıyla oluşturuldu! 🚀")
print(f"📂 Train: {len(train_files)} görüntü")
print(f"📂 Val: {len(val_files)} görüntü")
print(f"📂 Test: {len(test_files)} görüntü")

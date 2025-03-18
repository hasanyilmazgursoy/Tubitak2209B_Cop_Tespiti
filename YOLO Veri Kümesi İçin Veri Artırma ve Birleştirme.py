import cv2
import os
import glob
import numpy as np
import albumentations as A

# 📂 Orijinal ve artırılmış verilerin bulunduğu klasörlerin yolu
original_images_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\güncel data set\images"
original_labels_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\güncel data set\yolo_labels"
augmented_images_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\processed_data\augmented_images"
augmented_labels_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\processed_data\augmented_labels"

# 📂 Final veri kümesinin kaydedileceği klasörler
final_images_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\final_dataset\images"
final_labels_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\final_dataset\labels"

# 📂 Çıktı klasörlerini oluştur (eğer yoksa)
os.makedirs(final_images_dir, exist_ok=True)
os.makedirs(final_labels_dir, exist_ok=True)

# 🎨 Veri artırma işlemi için dönüşüm fonksiyonu tanımla
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # %50 ihtimalle yatay çevirme
    A.Rotate(limit=15, p=0.5),  # %50 ihtimalle ±15 derece döndürme
    A.RandomBrightnessContrast(p=0.5),  # %50 ihtimalle parlaklık ve kontrast değişimi
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

# 1️⃣ **Orijinal Görselleri ve Etiketleri Final Klasörüne Kopyala**
original_images = glob.glob(os.path.join(original_images_dir, "*.jpg"))  # 📂 Tüm JPG dosyalarını listele

for image_path in original_images:
    image_name = os.path.basename(image_path)  # 📌 Dosya adını al
    label_path = os.path.join(original_labels_dir, image_name.replace(".jpg", ".txt"))  # 🏷️ Etiket dosyasının yolu

    # 🖼️ OpenCV ile görüntüyü oku (OneDrive uyumlu)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"❌ Hata: {image_name} okunamadı!")  # 📌 Okuma hatası durumunda uyarı ver
        continue

    # 🖼️ Görseli final klasörüne kaydet
    final_image_path = os.path.join(final_images_dir, image_name)
    cv2.imencode('.jpg', image)[1].tofile(final_image_path)

    # 🏷️ Eğer etiket dosyası varsa, final klasörüne kopyala
    if os.path.exists(label_path):
        final_label_path = os.path.join(final_labels_dir, image_name.replace(".jpg", ".txt"))
        with open(label_path, "r") as f:
            lines = f.readlines()
        with open(final_label_path, "w") as f:
            f.writelines(lines)

print("✅ Orijinal veriler final_dataset klasörüne kopyalandı.")

# 2️⃣ **Veri Artırma İşlemi ve Yeni Verilerin Final Klasörüne Eklenmesi**
for image_path in original_images:
    image_name = os.path.basename(image_path)  # 📌 Görsel adını al
    label_path = os.path.join(original_labels_dir, image_name.replace(".jpg", ".txt"))  # 🏷️ Etiket dosya yolu

    # 🖼️ Görüntüyü oku (OneDrive uyumlu)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"❌ Hata: {image_name} okunamadı!")  # 📌 Okuma hatası kontrolü
        continue

    h, w, _ = image.shape  # 📏 Görselin genişlik ve yüksekliğini al

    # 🏷️ Eğer etiket dosyası varsa, oku
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        bboxes = []  # 📌 YOLO formatındaki bbox'ları saklamak için liste
        class_labels = []  # 📌 Sınıf etiketleri için liste

        for line in lines:
            data = line.strip().split()
            class_id = int(data[0])  # 🎯 Sınıf ID
            x_center, y_center, bbox_width, bbox_height = map(float, data[1:])  # 🔲 YOLO formatındaki bbox bilgileri

            # 📌 YOLO formatındaki verileri listelere ekle
            bboxes.append([x_center, y_center, bbox_width, bbox_height])
            class_labels.append(class_id)

        # 🎨 Veri artırma işlemini uygula
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image = augmented["image"]  # 🖼️ Yeni artırılmış görüntü
        augmented_bboxes = augmented["bboxes"]  # 🔲 Yeni artırılmış bbox'lar

        # 🆕 Yeni dosya adını belirle
        new_image_name = f"aug_{image_name}"
        new_label_name = f"aug_{image_name.replace('.jpg', '.txt')}"

        # 🖼️ Yeni artırılmış resmi final klasörüne kaydet
        output_image_path = os.path.join(final_images_dir, new_image_name)
        cv2.imencode('.jpg', augmented_image)[1].tofile(output_image_path)

        # 🏷️ Yeni artırılmış etiket dosyasını oluştur
        output_label_path = os.path.join(final_labels_dir, new_label_name)
        with open(output_label_path, "w") as f:
            for bbox, class_id in zip(augmented_bboxes, class_labels):
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

print("✅ Veri artırma işlemi tamamlandı ve final_dataset'e eklendi!")

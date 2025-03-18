import cv2
import os
import glob
import numpy as np

#Aşağıdaki kod tüm görüntüleri 640x640 boyutuna getirir ve yeni bir klasöre kaydeder. Etiketleri değiştirmez, sadece boyutları günceller.

# Klasör yolları
input_images_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\güncel data set\images"
input_labels_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\güncel data set\yolo_labels"
output_images_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\processed_data\resized_images"
output_labels_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\processed_data\resized_labels"

# Çıktı klasörlerini oluştur
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Hedef boyut
target_size = (640, 640)

# Görüntüleri işle
image_files = glob.glob(os.path.join(input_images_dir, "*.jpg"))

for image_path in image_files:
    image_name = os.path.basename(image_path)
    label_path = os.path.join(input_labels_dir, image_name.replace(".jpg", ".txt"))

    # Görüntüyü yükle
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"❌ Hata: {image_name} okunamadı!")
        continue

    h, w, _ = image.shape
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Yeni resim yoluna kaydet
    output_image_path = os.path.join(output_images_dir, image_name)
    cv2.imencode('.jpg', resized_image)[1].tofile(output_image_path)

    # Eğer etiket dosyası varsa, koordinatları güncelle
    if os.path.exists(label_path):
        output_label_path = os.path.join(output_labels_dir, image_name.replace(".jpg", ".txt"))

        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            data = line.strip().split()
            class_id = int(data[0])
            x_center, y_center, bbox_width, bbox_height = map(float, data[1:])

            # Yeni boyuta göre ölçekleme
            x_center *= target_size[0] / w
            y_center *= target_size[1] / h
            bbox_width *= target_size[0] / w
            bbox_height *= target_size[1] / h

            new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

        # Yeni etiketleri kaydet
        with open(output_label_path, "w") as f:
            f.writelines(new_lines)

print(f"✅ Tüm resimler {target_size} boyutuna getirildi ve yeni etiketler oluşturuldu!")

import cv2
import os
import glob
import numpy as np

# 📌 Dosya yolları (Final dataset'teki resim ve etiketler)
images_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\final_dataset\images"
labels_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\final_dataset\labels"
output_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\finalboxdeneme"  # ✅ Güncellenmiş çıktı klasörü

# 📌 Çıktı klasörünü oluştur
os.makedirs(output_dir, exist_ok=True)

# 📌 Çizim için renk ve kalınlık
color = (0, 255, 0)  # Yeşil (Bounding Box rengi)
thickness = 2  # Çizgi kalınlığı

# 📌 Görüntü dosyalarını al
image_files = glob.glob(os.path.join(images_dir, "*.jpg"))

for image_path in image_files:
    # 📌 Resim adını al (etiket dosyasını bulmak için)
    image_name = os.path.basename(image_path)
    label_path = os.path.join(labels_dir, image_name.replace(".jpg", ".txt"))

    # 📌 Eğer ilgili etiket dosyası yoksa atla
    if not os.path.exists(label_path):
        print(f"⚠️ Uyarı: {label_path} etiketi bulunamadı, atlanıyor.")
        continue

    # 📌 Alternatif yöntemle resmi oku (OneDrive hatalarına karşı)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    # 📌 Eğer resim okunamazsa hatayı yazdır ve devam et
    if image is None:
        print(f"❌ Hata: {image_path} okunamadı! Lütfen yolu kontrol et.")
        continue

    h, w, _ = image.shape  # 📌 Görüntü boyutlarını al
    print(f"✅ {image_name} başarıyla okundu! ({w}x{h})")

    # 📌 Etiketleri oku ve Bounding Box çiz
    with open(label_path, "r") as file:
        for line in file:
            data = line.strip().split()
            class_id = int(data[0])  # Sınıf etiketi (şimdilik kullanmıyoruz)
            x_center, y_center, bbox_width, bbox_height = map(float, data[1:])

            # 📌 YOLO formatındaki verileri piksel cinsine çevirme
            x_center *= w
            y_center *= h
            bbox_width *= w
            bbox_height *= h

            x1 = int(x_center - bbox_width / 2)
            y1 = int(y_center - bbox_height / 2)
            x2 = int(x_center + bbox_width / 2)
            y2 = int(y_center + bbox_height / 2)

            # 📌 Bounding box çiz
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # 📌 Çizilmiş resmi kaydet
    output_path = os.path.join(output_dir, image_name)
    cv2.imencode('.jpg', image)[1].tofile(output_path)  # ✅ Alternatif kaydetme yöntemi (OneDrive sorunu yaşamamak için)

print(f"🚀 Tüm görseller işlendi ve 'finalboxdeneme' klasörüne kaydedildi! ✅")

import cv2
import os
import glob
import numpy as np
import albumentations as A

# Giriş ve Çıkış Klasörleri
input_images_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\processed_data\enhanced_images"
output_images_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\processed_data\augmented_images"

# Eksik modülleri import et
os.makedirs(output_images_dir, exist_ok=True)

# Augmentation fonksiyonu
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # %50 ihtimalle yatay çevirme
    A.Rotate(limit=15, p=0.5),  # %50 ihtimalle döndürme
    A.RandomBrightnessContrast(p=0.5),  # Parlaklık & kontrast artırma
])

# Tüm görselleri işle
image_files = glob.glob(os.path.join(input_images_dir, "*.jpg"))

for image_path in image_files:
    image_name = os.path.basename(image_path)

    # Görseli oku
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"❌ Hata: {image_name} okunamadı!")
        continue

    # Augmentation uygula
    augmented = transform(image=image)["image"]

    # Yeni resim adı belirle
    output_image_path = os.path.join(output_images_dir, f"aug_{image_name}")
    cv2.imencode('.jpg', augmented)[1].tofile(output_image_path)

print("✅ Augmentation işlemi tamamlandı ve yeni resimler kaydedildi!")

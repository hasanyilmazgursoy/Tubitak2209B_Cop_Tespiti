import cv2
import os
import glob
import numpy as np

# Klasörler
input_images_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\processed_data\resized_images"
output_images_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\processed_data\enhanced_images"

# Çıktı klasörünü oluştur (import os unutulmuştu)
os.makedirs(output_images_dir, exist_ok=True)

# Görüntü kalitesini artırma fonksiyonu
def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Histogram eşitleme ile parlaklık artırma
    l = cv2.equalizeHist(l)
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_image

image_files = glob.glob(os.path.join(input_images_dir, "*.jpg"))

for image_path in image_files:
    image_name = os.path.basename(image_path)

    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"❌ Hata: {image_name} okunamadı!")
        continue

    enhanced_image = enhance_image(image)

    output_image_path = os.path.join(output_images_dir, image_name)
    cv2.imencode('.jpg', enhanced_image)[1].tofile(output_image_path)

print("✅ Tüm resimler parlaklık/kontrast iyileştirmesiyle kaydedildi!")

import os

# 📌 YOLO veri setinin ana klasörü
yolo_base_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\final_yolo_dataset"

# 📌 Train, Val, Test klasör yolları
images_train = os.path.join(yolo_base_dir, "images/train")
images_val = os.path.join(yolo_base_dir, "images/val")
images_test = os.path.join(yolo_base_dir, "images/test")

# 📌 data.yaml dosyasının yolu
yaml_path = os.path.join(yolo_base_dir, "data.yaml")

# 📌 Sınıf sayısını belirle (Senin datasetinde kaç sınıf varsa değiştir!)
num_classes = 1  # Örneğin, sadece "çöp" sınıfı varsa 1 yaz.

yaml_content = f"""train: {images_train}
val: {images_val}
test: {images_test}

nc: {num_classes}  # Sınıf sayısı
names: ["cop"]  # Sınıf isimleri
"""

# 📌 YAML dosyasını oluştur
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"✅ data.yaml dosyası oluşturuldu: {yaml_path}")

import json
import os

# Dosya yolları
annotations_path = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\dataset\annotations\annotations.json"
  # JSON dosyasının tam yolu
output_dir = r"C:\Users\sozce\OneDrive\Masaüstü\Yılmaz\yolo_labels"  # YOLO etiketlerinin kaydedileceği klasör

# Çıktı klasörünü oluştur
os.makedirs(output_dir, exist_ok=True)

# COCO formatındaki veriyi yükle
with open(annotations_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

# Resim bilgilerini içeren sözlük oluştur
image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

# YOLO formatına dönüştürme işlemi
for ann in coco_data["annotations"]:
    image_id = ann["image_id"]
    file_name = image_id_to_filename[image_id].replace(".jpg", ".txt")  # Etiket dosyası
    label_path = os.path.join(output_dir, file_name)

    # BBox (bounding box) bilgileri
    bbox = ann["bbox"]  # [x_min, y_min, width, height]
    category_id = ann["category_id"]

    # Kategoriyi YOLO formatına uygun hale getirme
    label_index = category_id  # Eğer tek bir sınıf kullanacaksan bunu 0 yapabilirsin.

    # Görsel boyutlarını al
    image_info = next(img for img in coco_data["images"] if img["id"] == image_id)
    image_width = image_info["width"]
    image_height = image_info["height"]

    # YOLO formatında normalleştirme
    x_center = (bbox[0] + bbox[2] / 2) / image_width
    y_center = (bbox[1] + bbox[3] / 2) / image_height
    width = bbox[2] / image_width
    height = bbox[3] / image_height

    # Etiket dosyasına yaz
    with open(label_path, "a", encoding="utf-8") as f:
        f.write(f"{label_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print(f"Dönüştürme tamamlandı! YOLO formatındaki dosyalar: {output_dir}")

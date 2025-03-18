import os

##Bu Python kodu, belirtilen dizindeki (label_dir) .txt uzantılı
# YOLO etiket dosyalarını kontrol ederek tekrar eden satırları kaldırır.
# Böylece, her etiketi yalnızca bir kez içeren temiz ve düzenli bir veri kümesi oluşturulur.
# Özellikle, aynı nesnenin birden fazla kez yanlışlıkla etiketlenmesini önlemek için faydalıdır.


label_dir = "/content/drive/My Drive/final_yolo_dataset/labels/val"

for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_dir, filename)

        with open(file_path, "r") as f:
            lines = f.readlines()

        unique_lines = list(set(lines))  # Tekrar edenleri kaldır

        if len(lines) != len(unique_lines):  # Değişiklik olmuşsa dosyayı güncelle
            with open(file_path, "w") as f:
                f.writelines(unique_lines)
            print(f"Düzenlendi: {filename}")
from ultralytics import YOLO
import os
import cv2
import glob

# YOLO modelini yükleyelim (önceden eğitilmiş model kullanılacak)
model = YOLO("yolov8n.pt")  # YOLOv8'in küçük boyutlu modeli

dataset_path = "Dataset/Nazugum"
output_folder = "Dataset/DataCollect"


# Çıktı klasörü yoksa oluştur
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Görüntüleri alalım
image_files = glob.glob(os.path.join(dataset_path, "*.jpg"))

# Her bir görüntüde kelebekleri tespit edip etiket dosyalarını oluşturuyoruz
for img_path in image_files:
    # Görüntüyü okuyalım
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # YOLO modelini kullanarak tespit yapalım
    results = model(img)

    # Her görüntü için bir .txt dosyası oluştur
    label_file_name = os.path.basename(img_path).replace(".jpg", ".txt")
    label_file_path = os.path.join(output_folder, label_file_name)

    # .txt dosyasını aç
    with open(label_file_path, "w") as f:
        for r in results:  # Birden fazla tespit olabilir
            boxes = r.boxes  # Tespit edilen bounding box'ları al
            for box in boxes:
                # Bounding box koordinatlarını ve sınıf ID'sini al
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # YOLO formatında normalize edilmiş koordinatları hesapla
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height

                # Sınıfı al (örneğin: 0)
                class_id = int(box.cls[0])

                # Etiketi dosyaya yaz (class_id x_center y_center bbox_width bbox_height)
                f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

    print(f"Etiket oluşturuldu: {label_file_path}")


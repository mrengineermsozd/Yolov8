from ultralytics import YOLO
import  cv2
from PIL import Image
#kütüphanelerin import edilmesi

model=YOLO('yolov8n.pt')  #yolov8 modelini yükle
cap = cv2.VideoCapture(0)  # Kamerayı başlat

while True:
    ret, frame = cap.read()
    # Görüntüyü flip yap (yatay çevirme)
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Kameradan görüntü alınamıyor.")
        break

    # Modeli çalıştır
    results = model.predict(source=frame, save=False, show=False)

    # Sonucu göster
    cizilmis_goruntu = results[0].plot()  # Sonuçları işaretli şekilde çiz
    cv2.imshow("YOLOv8", cizilmis_goruntu)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
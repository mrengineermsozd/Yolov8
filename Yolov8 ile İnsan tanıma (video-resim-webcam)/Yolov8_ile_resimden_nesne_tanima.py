from ultralytics import YOLO
from PIL import Image


model=YOLO('best.pt')

img1=Image.open("lamb.jpg")
sonuc=model.predict(source=img1,save=True)

import cv2
from ultralytics import YOLO
import pytesseract
import numpy as np
import matplotlib.pyplot as plt  # Matplotlib ekledik

# YOLOv8 modellerini yükle
plaka_model_paht = r"C:\Users\besse\coding\Aisoft Workspace\ultralytics\runs\detect\yolov8_plaka_model\weights\best.pt"
plaka_model = YOLO(plaka_model_paht)  # Kendi eğittiğiniz plaka tespit modeli
arac_model = YOLO("yolov8n.pt")  # Önceden eğitilmiş YOLOv8 modeli (araç tespiti için)

# Video dosyasını veya kamera akışını aç
video_path = r"C:\Users\besse\coding\YOLO\plaka\t3.mp4"  # Video dosyası veya 0 (kamera)
cap = cv2.VideoCapture(video_path)

# OCR için Tesseract ayarı (Windows için)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Plaka bilgilerini kaydetmek için dosya
output_file = "plaka_ve_arac_bilgileri.txt"
with open(output_file, "w") as f:
    f.write("Tespit Edilen Plakalar ve Araç Tipleri:\n")

# Görüntüleme yöntemini seç (OpenCV veya Matplotlib)
USE_OPENCV = False  # OpenCV kullanmak için True, Matplotlib için False yapın

while cap.isOpened():
    # Kareleri oku
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 ile plaka tespiti yap
    plaka_results = plaka_model(frame)

    # YOLOv8 ile araç tespiti yap
    arac_results = arac_model(frame)

    # Plaka tespiti sonuçlarını işle
    for result in plaka_results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Tespit kutuları
        confidences = result.boxes.conf.cpu().numpy()  # Güven skorları

        for box, confidence in zip(boxes, confidences):
            if confidence > 0.5:  # Güven skoru eşiği
                x1, y1, x2, y2 = map(int, box)  # Kutu koordinatları

                # Plakayı kırp (crop)
                plaka_img = frame[y1:y2, x1:x2]

                # Plakayı gri tonlamalı yap (OCR için daha iyi sonuç)
                plaka_gray = cv2.cvtColor(plaka_img, cv2.COLOR_BGR2GRAY)

                # OCR ile plaka metnini al
                plaka_metni = pytesseract.image_to_string(plaka_gray, config='--psm 8')
                plaka_metni = plaka_metni.strip()  # Boşlukları temizle

                # Plaka metnini ekrana yazdır
                print(f"Tespit Edilen Plaka: {plaka_metni}")

                # Plaka metnini dosyaya kaydet
                with open(output_file, "a") as f:
                    f.write(f"Plaka: {plaka_metni}\n")

                # Plaka etrafına kutu çiz ve metni yaz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Plaka: {plaka_metni}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Araç tespiti sonuçlarını işle
    for result in arac_results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Tespit kutuları
        confidences = result.boxes.conf.cpu().numpy()  # Güven skorları
        class_ids = result.boxes.cls.cpu().numpy()  # Sınıf ID'leri

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            if confidence > 0.5:  # Güven skoru eşiği
                x1, y1, x2, y2 = map(int, box)  # Kutu koordinatları
                arac_tipi = arac_model.names[int(class_id)]  # Sınıf adını al

                # Araç etrafına kutu çiz ve metni yaz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Araç: {arac_tipi}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Araç bilgisini dosyaya kaydet
                with open(output_file, "a") as f:
                    f.write(f"Araç Tipi: {arac_tipi}\n")

    # Görüntüyü göster
    if USE_OPENCV:
        # OpenCV ile görüntüleme
        cv2.imshow("Plaka ve Araç Tespiti", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşu ile çıkış
            break
    else:
        # Matplotlib ile görüntüleme
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # OpenCV BGR formatını RGB'ye çevir
        plt.title("Plaka ve Araç Tespiti")
        plt.axis('off')  # Eksenleri kapat
        plt.pause(0.01)  # Görüntüyü güncelle
        plt.clf()  # Önceki görüntüyü temizle

# Kaynakları serbest bırak
cap.release()
if USE_OPENCV:
    cv2.destroyAllWindows()
else:
    plt.close()  # Matplotlib penceresini kapat
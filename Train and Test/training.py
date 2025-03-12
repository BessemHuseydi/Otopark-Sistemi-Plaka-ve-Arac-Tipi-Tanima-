from ultralytics import YOLO

# Ana program bloğu
if __name__ == '__main__':
    # YOLOv9 model yapılandırmasını sıfırdan başlatın (varsayımsal olarak)
    model = YOLO("yolov8l.pt")  # Model yapılandırma dosyasını belirtin (yolov9.yaml)

    # Eğitim verilerini belirtin
    data_path = r"C:\Users\besse\coding\YOLO\plaka\License-Plate-Recognition-6\data.yaml"  # Eğitim verisi ve etiketlerin bulunduğu YAML dosyası

    # Eğitim parametrelerini tanımlayın
    remaining_epochs = 150  # Eğitim için kalan epoch sayısı

    # Modeli sıfırdan eğitmeye başla
    model.train(
        data=data_path,               # Eğitim verisi yolu (veri setinin yml dosyası)
        epochs=remaining_epochs,       # Eğitim için epoch sayısı
        patience=20,                   # Erken durdurma için sabır sayısı
        imgsz=640,                     # Resim boyutu (640x640 genellikle iyi sonuç verir)
        workers=8,                     # Paralel işçi sayısı
        batch=8,                       # Batch boyutu
        device=0,                      # Kullanılacak GPU cihazı (0, 1, 2, vb. numarası)
        name="yolov8_plaka_model",    # Modelin adı
        resume=False                   # Eğitim baştan başlasın
    )

    print("Eğitim tamamlandı!")

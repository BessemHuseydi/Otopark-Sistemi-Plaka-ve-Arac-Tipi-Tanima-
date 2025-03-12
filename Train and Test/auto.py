import os
from ultralytics import YOLO

def train_model():
    # YOLO modelini belirle
    model_name = "yolov8l.pt"  # İlk eğitim için başlangıç modeli
    trained_model_path = r"C:\Users\besse\coding\Aisoft Workspace\ultralytics\runs\detect\yolov8_plaka_model\weights\last.pt"  # En son eğitimden kaydedilen ağırlıklar

    # Eğer daha önce eğitim yapıldıysa, en son ağırlıkları kullan
    if os.path.exists(trained_model_path):
        print("[INFO] Daha önce eğitilmiş model bulundu. Eğitime kaldığı yerden devam ediliyor...")
        model = YOLO(trained_model_path)  # Önceki ağırlıkları yükle
        resume_training = True
    else:
        print("[INFO] Yeni eğitim başlatılıyor...")
        model = YOLO(model_name)  # Modeli sıfırdan yükle
        resume_training = False

    # Eğitim verisinin yolu
    data_path = r"C:\Users\besse\coding\YOLO\plaka\License-Plate-Recognition-6\data.yaml"

    # Eğitim parametreleri
    epochs = 150  # Toplam epoch sayısı
    patience = 20  # Erken durdurma için sabır süresi
    img_size = 640  # Görüntü boyutu
    batch_size = 8  # Batch boyutu
    workers = 8  # ÇOK ÖNEMLİ: Windows'ta multiprocessing hatası almamak için workers=0 yapıyoruz
    device = 0  # Kullanılacak GPU cihazı

    # Modeli eğit
    model.train(
        data=data_path,       
        epochs=epochs,        
        patience=patience,    
        imgsz=img_size,      
        workers=workers,  # Windows'ta hata almamak için 0 yaptık
        batch=batch_size,     
        device=device,        
        name="yolov8_plaka_model", 
        resume=resume_training  # Eğitime kaldığı yerden devam et
    )

    print("[INFO] Eğitim tamamlandı!")

# Ana program
if __name__ == '__main__':
    train_model()

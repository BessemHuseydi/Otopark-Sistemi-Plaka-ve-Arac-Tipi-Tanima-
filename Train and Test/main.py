import sys
import cv2
import time
import pytesseract
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
                             QWidget, QTableWidget, QTableWidgetItem, QMessageBox, QHBoxLayout)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

# Hazır araç tipi tespit modeli (YOLOv8) için model dosyasının yolunu belirtin:
vehicle_type_model_path = r"C:\Users\besse\coding\YOLO\Oto Park Sistemi\yolov8l.pt"
vehicle_type_model = YOLO(vehicle_type_model_path)

# Bu fonksiyon, tüm frame üzerinde araç tipini ve konumunu tespit eder.
def detect_vehicle_type(frame):
    results = vehicle_type_model(frame)
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            # İlk bulunan tespiti kullanıyoruz.
            box = result.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box[:4])
            cls_idx = int(result.boxes.cls[0].item())
            label = vehicle_type_model.model.names[cls_idx]
            return label, (x1, y1, x2, y2)
    return "Bilinmiyor", None

# Video işleme ve plaka tespiti için QThread sınıfı
class VideoThread(QThread):
    # Tespit edilen plaka bilgilerini (plaka text, araç tipi, plaka görüntüsü) ana pencereye aktarmak için sinyal
    plateDetected = pyqtSignal(str, str, np.ndarray)
    # Her karede üretilen görüntüyü (QImage formatında) ana pencereye aktarmak için sinyal
    changePixmap = pyqtSignal(QImage)

    def __init__(self, video_path, parent=None):
        super(VideoThread, self).__init__(parent)
        self.video_path = video_path
        self._run_flag = True
        # Plaka tespiti için YOLO modelini yükle; kendi model dosyanızın yolunu kullanın.
        plaka_model_path = r"C:\Users\besse\coding\YOLO\Oto Park Sistemi\model\best.pt"
        self.model = YOLO(plaka_model_path)
        # Aynı plakanın sürekli okunmaması için flag ve zaman bilgisi
        self.last_plate = None
        self.last_detection_time = time.time()
        self.reset_delay = 3  # plaka kaybolduktan kaç saniye sonra yeni tespit yapılacak
        self.ocr_wait_time = 1  # OCR okumaya başlamadan önce bekleme süresi (saniye)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break  # Videonun sonuna gelince çık
            current_time = time.time()
            # Eğer daha önce plaka tespit edilmişse ve reset süresi geçmişse, resetle
            if self.last_plate is not None and current_time - self.last_detection_time > self.reset_delay:
                self.last_plate = None

            # ---------------------------
            # 1) YOLO modelini kullanarak plaka tespiti yap
            # ---------------------------
            results = self.model(frame)
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box[:4])
                        # Sağ ve sol kenarlardan kırpma yapılıyor
                        plate_region1 = frame[y1:y2, x1:x2]
                        new_x1 = max(x1 + 10, 0)
                        new_x2 = min(x2 - 7, frame.shape[1])
                        plate_region = frame[y1:y2, new_x1:new_x2]
                        
                        # Plaka tespit edildikten sonra OCR okumadan önce bekleme süresi
                        time.sleep(self.ocr_wait_time)
                        
                        # OCR ile plaka okunuyor; --psm 7: tek satır metin için önerilir
                        plate_text = pytesseract.image_to_string(plate_region, config='--psm 7').strip()
                        if plate_text:
                            # Aynı plakanın tekrar okunmasını engellemek için kontrol
                            if self.last_plate is None or plate_text != self.last_plate:
                                self.last_plate = plate_text
                                self.last_detection_time = current_time
                                
                                # 2) Araç tipi tespiti: tüm frame üzerinde yapılan tespit
                                vehicle_type, vehicle_box = detect_vehicle_type(frame)
                                
                                # 3) Anotasyon: Plaka bölgesine dikdörtgen çizimi
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, plate_text, (new_x1, y1 - 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                
                                # 4) Araç tipi tespit edilebildiyse, mavi dikdörtgen çiziliyor.
                                if vehicle_box is not None:
                                    x1_v, y1_v, x2_v, y2_v = vehicle_box
                                    cv2.rectangle(frame, (x1_v, y1_v), (x2_v, y2_v), (255, 0, 0), 2)
                                    cv2.putText(frame, vehicle_type, (x1_v, y1_v - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                                
                                # 5) Sinyal gönder: plaka metni, araç tipi ve orijinal (kırpılmış) plaka görüntüsü
                                self.plateDetected.emit(plate_text, vehicle_type, plate_region1)

            # ---------------------------
            # 6) Kareyi PyQt5 arayüzüne aktarma
            # ---------------------------
            # OpenCV formatından QImage formatına çeviriyoruz
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # PyQt sinyaliyle ana pencereye gönder
            self.changePixmap.emit(qt_image)

        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self._run_flag = False
        self.wait()

# PyQt5 ana pencere sınıfı
class MainWindow(QMainWindow):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("Otopark Sistemi")
        self.setGeometry(100, 100, 1200, 600)  # Genişliği biraz artırdık

        # ---------------------------
        # 1) Arayüz Elemanları
        # ---------------------------
        # a) Video gösterimi için QLabel
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 360)  # İsteğe göre boyut
        self.video_label.setStyleSheet("background-color: black;")  # Arka plan

        # b) Tablonun sütun başlıkları
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Plaka", "Araç Tipi", "Giriş Zamanı", "Çıkış Zamanı", "Ücret (TL)", "Plaka Görüntüsü"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(5, 150)

        # c) Çıkış yap butonu
        self.exitButton = QPushButton("Çıkış Yap")
        self.exitButton.clicked.connect(self.register_exit)

        # ---------------------------
        # 2) Layout Ayarları
        # ---------------------------
        # Solda video, sağda tablo olacak şekilde yatay layout
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.video_label, 1)  # solda video
        h_layout.addWidget(self.table, 1)        # sağda tablo

        # Butonu alta eklemek için dikey layout
        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.exitButton)

        container = QWidget()
        container.setLayout(v_layout)
        self.setCentralWidget(container)

        # ---------------------------
        # 3) Video Thread başlatma
        # ---------------------------
        self.video_thread = VideoThread(video_path)
        # Plaka tespit sinyali -> tabloya ekleme fonksiyonu
        self.video_thread.plateDetected.connect(self.add_record)
        # Frame sinyali -> video_label güncelleme fonksiyonu
        self.video_thread.changePixmap.connect(self.updateImage)
        self.video_thread.start()

    def updateImage(self, qt_image):
        """VideoThread'den gelen QImage'ı QLabel üzerinde gösterir."""
        # Boyutu korumak veya ölçeklendirmek için scaled kullanabiliriz
        pixmap = QPixmap.fromImage(qt_image).scaled(self.video_label.width(),
                                                    self.video_label.height(),
                                                    Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def add_record(self, plate_text, vehicle_type, plate_image):
        # Yeni bir satır ekle: plaka, araç tipi, giriş zamanı, boş çıkış, boş ücret, plaka görüntüsü
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)
        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.table.setItem(row_position, 0, QTableWidgetItem(plate_text))
        self.table.setItem(row_position, 1, QTableWidgetItem(vehicle_type))
        self.table.setItem(row_position, 2, QTableWidgetItem(entry_time))
        self.table.setItem(row_position, 3, QTableWidgetItem(""))  # Çıkış zamanı henüz yok
        self.table.setItem(row_position, 4, QTableWidgetItem(""))  # Ücret henüz hesaplanmadı

        # Plaka görüntüsünü QImage -> QPixmap dönüştürüp tabloya ekleyelim
        if plate_image is not None and plate_image.size:
            plate_image_rgb = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
            height, width, channel = plate_image_rgb.shape
            bytes_per_line = channel * width
            q_img = QImage(plate_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(150, 80, Qt.KeepAspectRatio)
            label = QLabel()
            label.setPixmap(pixmap)
            self.table.setCellWidget(row_position, 5, label)

    def register_exit(self):
        # Seçili satırın çıkış zamanını ekle ve ücret hesapla
        selected = self.table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Uyarı", "Lütfen çıkış yapmak için bir kayıt seçiniz.")
            return
        row = selected[0].row()

        # Giriş zamanı alınır
        entry_time_str = self.table.item(row, 2).text()
        try:
            entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            QMessageBox.warning(self, "Hata", "Giriş zamanı formatı hatalı.")
            return

        exit_time = datetime.now()
        exit_time_str = exit_time.strftime("%Y-%m-%d %H:%M:%S")
        self.table.setItem(row, 3, QTableWidgetItem(exit_time_str))

        # Ücret hesaplama (örnek: saatlik 60 TL)
        duration = (exit_time - entry_time).total_seconds() / 3600
        fee = 60 * duration
        self.table.setItem(row, 4, QTableWidgetItem("{:.2f}".format(fee)))

    def closeEvent(self, event):
        # Pencere kapanırken video thread'ini durdur
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Video dosya yolunu girin:
    video_path = r"C:\Users\besse\coding\YOLO\Oto Park Sistemi\videos\v6.mp4"
    window = MainWindow(video_path)
    window.show()
    sys.exit(app.exec_())

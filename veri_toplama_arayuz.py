import sys
import os
import cv2
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class VeriToplamaArayuz(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mikro İfade Veri Toplama Arayüzü")
        self.setGeometry(200, 200, 600, 500)
        self.setStyleSheet("background-color: #f4f6fa;")
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        title = QLabel("Mikro İfade Veri Toplama")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #2d3e50;")
        layout.addWidget(title)

        # Tab widget oluştur
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #bdc3c7;
                border-radius: 10px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #ecf0f1;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #2980b9;
                color: white;
            }
        """)
        
        # Veri toplama tab'ı
        self.veri_toplama_tab = QWidget()
        self.veri_toplama_arayuzu_olustur()
        self.tab_widget.addTab(self.veri_toplama_tab, "📹 Veri Toplama")
        
        # Soru ekleme tab'ı
        self.soru_ekleme_tab = QWidget()
        self.soru_ekleme_arayuzu_olustur()
        self.tab_widget.addTab(self.soru_ekleme_tab, "➕ Soru Ekleme")
        
        layout.addWidget(self.tab_widget)

        self.status = QLabel("")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("font-size: 16px; color: #16a085;")
        layout.addWidget(self.status)

    def veri_toplama_arayuzu_olustur(self):
        """Veri toplama arayüzünü oluşturur"""
        layout = QVBoxLayout(self.veri_toplama_tab)
        layout.setSpacing(20)
        
        # Mevcut soruları göster
        self.mevcut_sorular_label = QLabel("Mevcut Sorular:")
        self.mevcut_sorular_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(self.mevcut_sorular_label)
        
        self.mevcut_sorular_list = QListWidget()
        self.mevcut_sorular_list.setMaximumHeight(100)
        self.mevcut_sorular_list.setStyleSheet("""
            QListWidget {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 5px;
                background-color: white;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)
        layout.addWidget(self.mevcut_sorular_list)
        
        # Mevcut soruları yükle
        self.mevcut_sorulari_yukle()

        form = QFormLayout()
        self.kullanici = QLineEdit()
        self.kullanici.setPlaceholderText("Kullanıcı adı veya ID")
        self.kullanici.setText("1")  # Varsayılan
        self.soru = QSpinBox()
        self.soru.setRange(1, 100)
        self.etiket = QComboBox()
        self.etiket.addItems(["dogru", "yalan"])
        self.sure = QSpinBox()
        self.sure.setRange(1, 10)
        self.sure.setValue(3)
        
        form.addRow("Kullanıcı:", self.kullanici)
        form.addRow("Soru No:", self.soru)
        form.addRow("Etiket:", self.etiket)
        form.addRow("Kayıt Süresi (sn):", self.sure)
        layout.addLayout(form)

        self.kayit_btn = QPushButton("📹 Kamera ile Kayıt Başlat")
        self.kayit_btn.setStyleSheet("""
            QPushButton { 
                background-color: #2980b9; 
                color: white; 
                font-size: 18px; 
                font-weight: bold; 
                border-radius: 8px; 
                padding: 15px 30px; 
            } 
            QPushButton:hover { 
                background-color: #2471a3; 
            }
        """)
        self.kayit_btn.setCursor(Qt.PointingHandCursor)
        self.kayit_btn.clicked.connect(self.kayit_baslat)
        layout.addWidget(self.kayit_btn, alignment=Qt.AlignCenter)

    def soru_ekleme_arayuzu_olustur(self):
        """Soru ekleme arayüzünü oluşturur"""
        layout = QVBoxLayout(self.soru_ekleme_tab)
        layout.setSpacing(20)
        
        # Başlık
        baslik = QLabel("Yeni Soru Ekleme")
        baslik.setAlignment(Qt.AlignCenter)
        baslik.setStyleSheet("font-size: 20px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(baslik)
        
        # Soru formu
        form = QFormLayout()
        
        self.yeni_soru_no = QSpinBox()
        self.yeni_soru_no.setRange(1, 100)
        self.yeni_soru_no.setValue(16)  # Varsayılan
        
        self.yeni_soru_metni = QTextEdit()
        self.yeni_soru_metni.setMaximumHeight(100)
        self.yeni_soru_metni.setPlaceholderText("Soru metnini buraya yazın...")
        
        form.addRow("Soru No:", self.yeni_soru_no)
        form.addRow("Soru Metni:", self.yeni_soru_metni)
        layout.addLayout(form)
        
        # Örnek sorular
        ornek_sorular_label = QLabel("Örnek Sorular:")
        ornek_sorular_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(ornek_sorular_label)
        
        ornek_sorular = [
            "Hiç birini kırdığın için kendini suçlu hissettin mi?",
            "Kendini olduğundan daha iyi göstermeye çalıştığın olur mu?",
            "Kimseyle paylaşamadığın bir sırrın var mı?",
            "En yakınlarına bile yalan söylediğin oldu mu?",
            "Zaman zaman kendinden nefret ettiğin oldu mu?"
        ]
        
        for i, soru in enumerate(ornek_sorular, 1):
            ornek_btn = QPushButton(f"{i}. {soru[:50]}...")
            ornek_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ecf0f1;
                    border: 1px solid #bdc3c7;
                    border-radius: 5px;
                    padding: 8px;
                    text-align: left;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #d5dbdb;
                }
            """)
            ornek_btn.clicked.connect(lambda checked, s=soru: self.yeni_soru_metni.setText(s))
            layout.addWidget(ornek_btn)

        self.soru_ekle_btn = QPushButton("➕ Soru Ekle")
        self.soru_ekle_btn.setStyleSheet("""
            QPushButton { 
                background-color: #27ae60; 
                color: white; 
                font-size: 16px; 
                font-weight: bold; 
                border-radius: 8px; 
                padding: 12px 25px; 
            } 
            QPushButton:hover { 
                background-color: #229954; 
            }
        """)
        self.soru_ekle_btn.setCursor(Qt.PointingHandCursor)
        self.soru_ekle_btn.clicked.connect(self.soru_ekle)
        layout.addWidget(self.soru_ekle_btn, alignment=Qt.AlignCenter)

    def mevcut_sorulari_yukle(self):
        """Mevcut soruları listeler"""
        try:
            # CSV dosyasından soruları yükle
            if os.path.exists('psikolojik_sorular_yeni.csv'):
                df = pd.read_csv('psikolojik_sorular_yeni.csv')
                self.mevcut_sorular_list.clear()
                for i, soru in enumerate(df['soru'], 1):
                    self.mevcut_sorular_list.addItem(f"Soru {i}: {soru[:50]}...")
            else:
                self.mevcut_sorular_list.addItem("Henüz soru bulunmuyor. Önce soru ekleyin.")
        except Exception as e:
            self.mevcut_sorular_list.addItem(f"Sorular yüklenirken hata: {e}")

    def kayit_baslat(self):
        kullanici_id = self.kullanici.text().strip()
        soru_id = self.soru.value()
        etiket = self.etiket.currentText()
        sure = self.sure.value()
        
        if not kullanici_id:
            QMessageBox.warning(self, "Uyarı", "Kullanıcı adı giriniz!")
            return
            
        klasor = f"dataset/{kullanici_id}/soru{soru_id}_{etiket}"
        os.makedirs(klasor, exist_ok=True)
        
        # Kamera başlat
        kamera = cv2.VideoCapture(0)
        if not kamera.isOpened():
            QMessageBox.critical(self, "Hata", "Kamera başlatılamadı!")
            return
            
        sayac = 0
        self.status.setText("Kayıt başlıyor...")
        self.kayit_btn.setEnabled(False)
        
        while sayac < sure * 30:
            ret, frame = kamera.read()
            if not ret:
                break
            dosya_yolu = os.path.join(klasor, f"frame_{sayac:03}.jpg")
            cv2.imwrite(dosya_yolu, frame)
            sayac += 1
            cv2.imshow("Kayıt", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        kamera.release()
        cv2.destroyAllWindows()
        
        self.status.setText(f"✅ Kayıt tamamlandı: {klasor} ({sayac} frame)")
        self.kayit_btn.setEnabled(True)

    def soru_ekle(self):
        soru_no = self.yeni_soru_no.value()
        soru_metni = self.yeni_soru_metni.toPlainText().strip()
        
        if not soru_metni:
            QMessageBox.warning(self, "Uyarı", "Soru metni giriniz!")
            return
            
        try:
            # CSV dosyasını oku veya oluştur
            if os.path.exists('psikolojik_sorular_yeni.csv'):
                df = pd.read_csv('psikolojik_sorular_yeni.csv')
            else:
                df = pd.DataFrame(columns=['soru'])
            
            # Yeni soruyu ekle
            yeni_satir = pd.DataFrame({'soru': [soru_metni]})
            df = pd.concat([df, yeni_satir], ignore_index=True)
            
            # CSV'ye kaydet
            df.to_csv('psikolojik_sorular_yeni.csv', index=False)
            
            QMessageBox.information(self, "Başarılı", f"Soru {soru_no} başarıyla eklendi!")
            
            # Formu temizle
            self.yeni_soru_metni.clear()
            self.yeni_soru_no.setValue(soru_no + 1)
            
            # Mevcut soruları yenile
            self.mevcut_sorulari_yukle()
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Soru eklenirken hata oluştu: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VeriToplamaArayuz()
    window.show()
    sys.exit(app.exec_()) 
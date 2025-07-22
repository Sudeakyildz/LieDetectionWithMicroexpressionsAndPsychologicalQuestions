# Mikro İfade Yalan Tespit Sistemi

Bu proje, psikolojik sorular ve mikro ifade analizi kullanarak yalan tespiti yapan bir uygulamadır.

## Özellikler

- Kullanıcı bilgileri girişi (ad-soyad, yaş, cinsiyet, meslek)
- Kamera ile mikro ifade analizi
- 15 psikolojik soru ile test
- Gerçek zamanlı duygu analizi
- Detaylı test sonuç raporu
- Dataset karşılaştırmalı analiz
- **Yeni: Dataset yüz ifadesi analizi** - Dataset'teki görüntülerle doğrudan karşılaştırma

## Analiz Yöntemleri

1. **Psikolojik Analiz** (%25 ağırlık) - Soru-cevap psikolojisine dayalı
2. **Mikro İfade Analizi** (%30 ağırlık) - Gerçek zamanlı duygu analizi
3. **Dataset Karşılaştırması** (%25 ağırlık) - Referans dağılımlarla karşılaştırma
4. **Dataset Yüz İfadesi Analizi** (%20 ağırlık) - Dataset görüntüleriyle doğrudan karşılaştırma

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Uygulamayı çalıştırın:
```bash
python main.py
```

## Kullanım

1. Uygulama başladığında kullanıcı bilgilerinizi girin
2. "Test Başlat" butonuna tıklayın
3. Kamera açılacak ve sorular gösterilmeye başlayacak
4. Her soru için "Evet" veya "Hayır" cevabını verin
5. Tüm sorular bittiğinde "Sonuçları Göster" butonuna tıklayın
6. Sonuç ekranında detaylı analiz raporunu görüntüleyin

## Sonuç Raporu

Her soru için şu analizler yapılır:
- **Psikolojik Analiz**: Soru tipine göre beklenen davranış
- **Mikro İfade Analizi**: Gerçek zamanlı duygu dağılımı
- **Dataset Karşılaştırması**: Referans verilerle benzerlik
- **Dataset Yüz İfadesi Analizi**: Doğruluk/yalan yüzdesi
- **Genel Sonuç**: 4 analizin ağırlıklı ortalaması

## Gereksinimler

- Python 3.8 veya üzeri
- Webcam
- İnternet bağlantısı (DeepFace modeli için)

## Notlar

- Test sırasında iyi aydınlatılmış bir ortamda olun
- Kameraya doğrudan bakın
- Doğal davranın ve rahat olun
- Dataset yüz ifadesi analizi için dataset klasöründeki görüntüler kullanılır 